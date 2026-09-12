# Moonshine Streaming ASR

[Moonshine](https://huggingface.co/moonshine-ai) is an English streaming ASR
family. audio.cpp supports the upstream tiny, small, and medium streaming
checkpoints through the `moonshine_asr` family.

| Field | Value |
|---|---|
| Family | `moonshine_asr` |
| Task | `asr` |
| Modes | `offline`, `streaming` |
| Language | English |
| Recommended package | `moonshine_streaming_tiny_q8_0` |
| Output | Transcript text |
| Timestamps | Not exposed |

## Install

The recommended packages are standalone Q8_0 GGUFs in the aggregate audio.cpp
GGUF repo:

```bash
python3 tools/model_manager_v2.py install moonshine_streaming_tiny_q8_0
python3 tools/model_manager_v2.py install moonshine_streaming_small_q8_0
python3 tools/model_manager_v2.py install moonshine_streaming_medium_q8_0
```

The GGUFs embed the model spec, configuration, and tokenizer sidecars, so each
file can be loaded directly.

## CLI

Offline transcription:

```bash
audiocpp_cli --task asr \
  --family moonshine_asr \
  --model models/Moonshine-Streaming-GGUF/moonshine-streaming-tiny-q8_0.gguf \
  --backend cuda \
  --audio assets/resources/sample_16k.wav \
  --text-out transcript.txt \
  --log
```

Streaming session with file input:

```bash
audiocpp_cli --task asr --mode streaming \
  --family moonshine_asr \
  --model models/Moonshine-Streaming-GGUF/moonshine-streaming-tiny-q8_0.gguf \
  --backend cuda \
  --audio assets/resources/sample_16k.wav \
  --text-out transcript.txt \
  --log
```

## Options

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--audio` | WAV path | required | Speech input. |
| `--max-tokens` | integer | audio-duration derived | Maximum generated transcript tokens. |
| `--text-out` | TXT path | not set | Transcript output. |
| `--session-option moonshine_asr.weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Shared matmul weight storage type. |
| `--session-option moonshine_asr.encoder_gelu=<mode>` | `erf`, `exact`, `tanh`, `quick` | `quick` | Encoder GELU lowering. |
| `--session-option moonshine_asr.cpu_blas_scheduler=true\|false` | bool | `true` | Use BLAS/Accelerate for supported CPU encoder matmuls. |

## Convert

The helper converts the upstream safetensors checkpoints into standalone GGUFs:

```bash
python3 tools/community_models/convert_moonshine_asr.py \
  --checkpoint models/moonshine-streaming-tiny \
  --converter build/debug/bin/audiocpp_gguf \
  --output models/Moonshine-Streaming-GGUF/moonshine-streaming-tiny-q8_0.gguf \
  --type q8_0 \
  --overwrite
```

Use the same command shape for `moonshine-streaming-small` and
`moonshine-streaming-medium`.

## Sources

- Tiny: <https://huggingface.co/moonshine-ai/moonshine-streaming-tiny>
- Small: <https://huggingface.co/moonshine-ai/moonshine-streaming-small>
- Medium: <https://huggingface.co/moonshine-ai/moonshine-streaming-medium>
