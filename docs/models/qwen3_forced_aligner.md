# Qwen3 Forced Aligner

For shared conversion guidance, see
[Qwen3 GGUF checkpoints](qwen3.md#gguf-checkpoints).

The forced aligner maps an exact transcript onto speech audio. It is not an ASR route: the transcript is required input.
Standalone forced alignment does not chunk audio because exact transcript/audio
chunk pairing cannot be inferred safely from a raw transcript. For long-audio
timestamping, use Qwen3 ASR with `--words-out` and
`qwen3_asr.forced_aligner_path`; ASR chunks the audio first, then aligns
each recognized transcript to its matching audio chunk.

| Field | Value |
|---|---|
| Family | `qwen3_forced_aligner` |
| GGUF model | `models/Qwen3-ForcedAligner-0.6B-GGUF/qwen3-forced-aligner-0.6b-q8_0.gguf` |
| Task | `align` |
| Modes | `offline` |
| Input | Speech WAV plus exact transcript |
| Output | Word timestamp JSON through `--words-out` |

```bash
audiocpp_cli --task align --family qwen3_forced_aligner --model models/Qwen3-ForcedAligner-0.6B-GGUF/qwen3-forced-aligner-0.6b-q8_0.gguf --backend cuda --audio assets/resources/sample_16k.wav --text "Some call me nature, others call me Mother Nature." --language English --words-out words.json
```

## Common Options (use directly)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--audio` | WAV path | required | Speech audio at any sample rate; timestamps are reported in that file's sample domain. |
| `--text` | exact transcript | required | Transcript to align. |
| `--language` | language code | required | Transcript language. |
| `--audio-chunk-mode` | `auto`, `none` | `auto` | Standalone forced alignment runs one audio/transcript pair. `fixed` and `vad` are rejected because transcript chunk boundaries would be ambiguous. |
| `--words-out` | JSON path | not set | Word timestamp output. |

## Request Options (use with `--request-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `clamp_timestamps_to_audio` | bool | `false` | Opt-in guard for word timestamps: keep repaired spans inside the local audio input. Default preserves existing timestamp repair behavior. |

## Session Options (use with `--session-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `qwen3_forced_aligner.weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Aligner thinker weight type. |
