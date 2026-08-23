# Speech Analysis

| Model | Family | Task | Quick Start |
|---|---|---|---|
| Silero VAD | `silero_vad` | `vad` | [Silero VAD](#silero-vad) |
| MarbleNet VAD | `marblenet_vad` | `vad` | [MarbleNet VAD](#marblenet-vad) |
| Sortformer Diarization | `sortformer_diar` | `diar` | [Sortformer Diarization](#sortformer-diarization) |
| MMS Forced Aligner | `mms_forced_aligner` | `align` | [MMS Forced Aligner](#mms-forced-aligner) |

This page covers VAD, diarization, and forced-aligner models. ASR models are documented in [ASR models](asr.md).

Common CLI shape:

```bash
audiocpp_cli --task <task> --family <family> --model <model-dir> --backend cuda --audio <audio.wav> ...
```

When `--mode streaming` is used, the selected model provides its default streaming policy.

## Silero VAD

Silero VAD is bundled as a small framework asset and detects speech segments. It supports offline and streaming modes.

| Field | Value |
|---|---|
| Family | `silero_vad` |
| Model directory | `assets/framework/models/silero_vad` |
| Task | `vad` |
| Modes | `offline`, `streaming` |
| Output | Speech segment JSON through `--segments-out`; offline VAD chunk windows through `--vad-chunks-out` |
| Sample rates | 16 kHz path is used by the examples; 512-sample streaming chunks are required by the model path |

Offline:

```bash
audiocpp_cli --task vad --family silero_vad --model assets/framework/models/silero_vad --backend cuda --audio assets/resources/sample_16k.wav --segments-out segments.json
```

Offline VAD chunk planning:

```bash
audiocpp_cli \
  --task vad \
  --family silero_vad \
  --model assets/framework/models/silero_vad \
  --backend cuda \
  --audio assets/resources/sample_16k.wav \
  --segments-out segments.json \
  --vad-chunks-out vad_chunks.json \
  --vad-chunk-max-seconds 45 \
  --vad-chunk-merge-gap-seconds 0.5 \
  --vad-chunk-padding-seconds 0.25
```

Streaming:

```bash
audiocpp_cli --task vad --family silero_vad --model assets/framework/models/silero_vad --backend cuda --mode streaming --audio <512-sample-16k-wav> --segments-out segments.json
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--audio` | WAV path | required | Input audio. |
| `--mode` | `offline`, `streaming` | `offline` | Full-file or streaming VAD. |
| `--segments-out` | JSON path | not set | Write speech segments. |
| `--vad-chunks-out` | JSON path | not set | Write offline VAD-based chunk windows. |
| `--vad-chunk-max-seconds` | seconds | `45` | Maximum VAD chunk length. |
| `--vad-chunk-merge-gap-seconds` | seconds | `0.5` | Merge padded speech spans separated by this gap or less. |
| `--vad-chunk-padding-seconds` | seconds | `0.25` | Pad each speech segment before chunk planning. |
| `--request-option threshold=<float>` | float | `0.5` | Speech probability threshold. |
| `--request-option neg_threshold=<float>` | float | `threshold - 0.15`, clamped to at least `0.01` | Negative threshold used by the state machine when not set directly. |
| `--request-option min_speech_duration_ms=<n>` | integer ms | `250` | Minimum speech duration. |
| `--request-option min_silence_duration_ms=<n>` | integer ms | `100` | Minimum silence duration. |
| `--request-option speech_pad_ms=<n>` | integer ms | `30` | Padding around speech segments. |
| `--request-option max_speech_duration_s=<float>` | seconds | `1000000000` | Maximum speech segment length. |

## MarbleNet VAD

MarbleNet VAD is an offline speech activity detector.

| Field | Value |
|---|---|
| Family | `marblenet_vad` |
| Model directory | `assets/framework/models/marblenet_vad` |
| Task | `vad` |
| Modes | `offline` |
| Output | Speech segment JSON through `--segments-out` |
| Streaming | Not exposed |

```bash
audiocpp_cli --task vad --family marblenet_vad --model assets/framework/models/marblenet_vad --backend cuda --audio speech_16k.wav --segments-out segments.json
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--audio` | WAV path | required | Input audio. |
| `--segments-out` | JSON path | not set | Write speech segments. |
| `--request-option threshold=<float>` | float | `0.5` | Speech probability threshold. |

## Sortformer Diarization

Sortformer diarization identifies speaker turns. The packaged model path is the 4-speaker variant.

| Field | Value |
|---|---|
| Family | `sortformer_diar` |
| Model directory | `models/Sortformer-Diar-4spk-v1-GGUF` |
| Task | `diar` |
| Modes | `offline` |
| Output | Speaker turn JSON through `--turns-out` |
| Speakers | Up to the speaker count supported by the model package; the default model is 4-speaker |

```bash
audiocpp_cli --task diar --family sortformer_diar --model models/Sortformer-Diar-4spk-v1-GGUF/sortformer-diar-4spk-v1-q8_0.gguf --backend cuda --audio meeting_16k.wav --turns-out turns.json
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--audio` | WAV path | required | Meeting or conversation audio. |
| `--turns-out` | JSON path | not set | Write speaker turns. |
| `--request-option speaker_threshold=<float>` | float | session default | Per-request speaker activation threshold. |
| `--request-option speaker_min_frames=<n>` | integer | session default | Per-request minimum speaker segment frames. |
| `--request-option speaker_pad_frames=<n>` | integer | session default | Per-request padding around speaker turns. |
| `--session-option sortformer_diar.speaker_threshold=<float>` | float | `0.5` | Default speaker activation threshold. |
| `--session-option sortformer_diar.speaker_min_frames=<n>` | integer | `0` | Default minimum speaker segment frames. |
| `--session-option sortformer_diar.speaker_pad_frames=<n>` | integer | `0` | Default padding around speaker turns. |
| `--session-option sortformer_diar.session_len_sec=<float>` | seconds | `20.0` | Diarization graph window length. |
| `--session-option sortformer_diar.graph_capacity_mode=<mode>` | `fixed`, `tiered`, `grow`, `double` | backend default | Offline graph capacity policy. |
| `--session-option sortformer_diar.graph_arena_mb=<n>` | MB | `512` | Inference graph arena size. |
| `--session-option sortformer_diar.weight_context_mb=<n>` | MB | `128` | Weight context size. |
| `--session-option sortformer_diar.weight_type=<type>` | storage type | `f32` | Default weight storage type. |
| `--session-option sortformer_diar.matmul_weight_type=<type>` | storage type | `weight_type` | Matmul weight storage override. |
| `--session-option sortformer_diar.conv_weight_type=<type>` | storage type | `weight_type` | Convolution weight storage override. |

Compatibility aliases are applied before v1 option validation:

| Legacy session option | v1 session option |
|---|---|
| `speaker_threshold` | `sortformer_diar.speaker_threshold` |
| `speaker_min_frames` | `sortformer_diar.speaker_min_frames` |
| `speaker_pad_frames` | `sortformer_diar.speaker_pad_frames` |
| `session_len_sec` | `sortformer_diar.session_len_sec` |
| `graph_context_mb`, `sortformer_diar.graph_context_mb` | `sortformer_diar.graph_arena_mb` |
| `graph_capacity_mode`, `offline_graph_capacity_mode` | `sortformer_diar.graph_capacity_mode` |
| `weight_context_mb` | `sortformer_diar.weight_context_mb` |
| `weight_type` | `sortformer_diar.weight_type` |
| `matmul_weight_type` | `sortformer_diar.matmul_weight_type` |
| `conv_weight_type` | `sortformer_diar.conv_weight_type` |

For backend weight-type controls, use `audiocpp_cli --inspect --model <model-dir> --family <family>`.

## MMS Forced Aligner

The MMS forced aligner aligns an exact transcript to mono/stereo audio and
returns per-word start/end timestamps. Native normalization covers Dutch and
English; a pre-romanized mode accepts ASCII romanization for other languages.

| Field | Value |
|---|---|
| Family | `mms_forced_aligner` |
| Model directory | `models/mms-300m-1130-forced-aligner` (safetensors) or a local GGUF |
| Task | `align` |
| Mode | `offline` only |
| Output | `word_timestamps` |

```bash
audiocpp_cli --task align --family mms_forced_aligner --model models/mms-300m-1130-forced-aligner \
  --audio speech.wav --text "The quick brown fox." --language eng --words-out words.json
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--audio` | WAV path | required | Input audio. |
| `--text` | string | required | Exact transcript to align. |
| `--language` | `nl`, `nld`, `en`, `eng` (latin); any code (pre_romanized) | required | Transcript language. |
| `--request-option text_normalization=<mode>` | `latin`, `pre_romanized` | `latin` | Native Latin normalization or caller-supplied romanization. |
| `--request-option star_frequency=<mode>` | `segment`, `edges` | `segment` | Virtual `<star>` target placement. |
| `--request-option merge_threshold_sec=<float>` | float >= 0 | `0.0` | Merge words whose gap is at or below this many seconds. |
| `--words-out` | JSON path | not set | Write per-word timestamps. |

The checkpoint is CC-BY-NC-4.0 and GGUF conversion is local-only. See the
[full model guide](community_models/mms_forced_aligner.md) for install,
conversion, licensing, and boundary-parity evidence.
