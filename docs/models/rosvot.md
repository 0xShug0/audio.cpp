# ROSVOT

ROSVOT transcribes singing audio into musical notes. It predicts note pitches,
boundaries and word-to-note alignment; it does not synthesize singing or transcribe
lyrics. The package includes ROSVOT, the RWBD word-boundary predictor and RMVPE
pitch extraction. Inference is offline.

## Download

The F32 package is available in the standalone
[audio-cpp/SoulX-Singer-GGUF](https://huggingface.co/audio-cpp/SoulX-Singer-GGUF/tree/main/ROSVOT-GGUF)
repository. Download `ROSVOT-GGUF/rosvot-f32.gguf`.

## Convert

Download the `rosvot/` directory from
[SoulX-Singer-Preprocess](https://huggingface.co/Soul-AILab/SoulX-Singer-Preprocess/tree/main/rosvot).
Its three subdirectories, `rosvot`, `rwbd` and `rmvpe`, must remain together.
Build `audiocpp_gguf`, then convert:

```bash
python tests/rosvot/convert_gguf.py \
  --source models/ROSVOT/rosvot \
  --staging models/ROSVOT/staging \
  --output models/ROSVOT-GGUF/rosvot-f32.gguf \
  --log convert-rosvot.log
```

The GGUF embeds all three checkpoints, configuration, model contract and notices.
No Python dependencies are needed for native inference.

## Run

```bash
audiocpp_cli --task midi --family rosvot \
  --model models/ROSVOT-GGUF/rosvot-f32.gguf \
  --backend cuda --threads 8 --audio singing.wav --out notes.mid --log
```

For JSON output, add `--request-option output_format=json` and use `--out notes.json`.
JSON notes contain MIDI `pitch` (zero denotes a rest), `start` and `end` in seconds,
and one-based `word_index`. Predicted word boundaries do not contain lyric text.
MIDI omits rests. Use a clean isolated vocal for the clearest interpretation;
separation is a separate task, not implicitly run by this family.

## Options

| Request option | Default | Meaning |
|---|---|---|
| `output_format` | `midi` | `midi` or `json` |
| `note_boundary_threshold` | `0.85` | Note-boundary probability threshold, in [0, 1] |
| `word_boundary_threshold` | `0.5` | RWBD probability threshold, in [0, 1] |
| `word_durations_file` | Omitted | JSON array of word durations in seconds; skips RWBD |

The supplied word durations must describe the input audio, including pauses.
They provide timing, not lyrics. Audio is mixed to mono and resampled to 24 kHz.
A request is limited to approximately 160 seconds, matching the upstream single-item
frame limit. Streaming is not supported by the reference model.

## License

ROSVOT code is MIT licensed. Checkpoint provenance and notices are included with
the conversion tests and embedded package. See `tests/rosvot/NOTICE`.
