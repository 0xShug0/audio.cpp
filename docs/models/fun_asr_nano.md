# Fun-ASR-Nano

audio.cpp runs Fun-ASR-Nano-2512 as a native offline ASR model on CPU and CUDA.
The runtime accepts the official Hugging Face safetensors checkpoint and
standalone F16 or Q8_0 GGUF files.

| Field | Value |
|---|---|
| Family | `fun_asr_nano` |
| Task | `asr` |
| Mode | `offline` |
| Languages | `auto`, `zh`, `en`, `ja` |
| Audio | WAV; converted to mono 16 kHz internally |
| Output | Transcript text |
| Streaming | Not exposed |
| Timestamps | Not exposed |

## Install

The model-spec manager installs the Q8_0 standalone GGUF by default:

```bash
python3 tools/model_manager_v2.py install fun_asr_nano
```

Select another published format explicitly:

```bash
python3 tools/model_manager_v2.py install fun_asr_nano_2512_f16
python3 tools/model_manager_v2.py install fun_asr_nano_2512_safetensors
```

Published GGUF files and their SHA256 manifest are available at
[FunAudioLLM/Fun-ASR-Nano-2512-GGUF](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512-GGUF).
The package catalog pins an immutable Hub revision.

## CLI

```bash
audiocpp_cli \
  --task asr \
  --family fun_asr_nano \
  --model models/Fun-ASR-Nano-2512-GGUF/fun-asr-nano-2512-q8_0.gguf \
  --backend cuda \
  --audio speech.wav \
  --text-out transcript.txt
```

The same command accepts the F16 GGUF file or the official safetensors model
directory.

### Request Options

| Option | Values | Default | Meaning |
|---|---|---|---|
| `language` | `auto`, `zh`, `en`, `ja` | `auto` | Recognition language hint. |
| `enable_itn` | `true`, `false` | `true` | Enable inverse text normalization in the prompt. |
| `max_tokens` | positive integer | `512` | Maximum transcript tokens generated per chunk. |
| `audio_chunk_mode` | `auto`, `fixed`, `none` | `auto` | Offline audio chunking policy. |
| `audio_chunk_seconds` | positive seconds | `30` | Chunk duration for fixed/automatic chunking. |

For example:

```bash
audiocpp_cli --task asr --family fun_asr_nano \
  --model models/Fun-ASR-Nano-2512-GGUF/fun-asr-nano-2512-q8_0.gguf \
  --backend cuda --audio meeting.wav \
  --language zh \
  --request-option enable_itn=true \
  --audio-chunk-mode fixed \
  --audio-chunk-seconds 30
```

### Weight Storage

`fun_asr_nano.weight_type` sets a shared storage preference. Component overrides are
`fun_asr_nano.encoder_weight_type`, `fun_asr_nano.adaptor_weight_type`, and
`fun_asr_nano.decoder_weight_type`. Supported values are `native`, `f32`,
`f16`, `bf16`, and `q8_0`.

On CUDA, native, F16, and Q8 shared preferences are promoted to BF16 for the
decoder to keep logits stable. Encoder and adaptor weights retain the shared
type, so a Q8_0 GGUF keeps those large matrix paths quantized. An explicit
`fun_asr_nano.decoder_weight_type` forces that component's storage type. CPU
uses the requested or native stored types without promotion.

## Server

`server.json`:

```json
{
  "host": "127.0.0.1",
  "port": 8080,
  "backend": "cuda",
  "device": 0,
  "threads": 4,
  "lazy_load": true,
  "models": [
    {
      "id": "fun-asr-nano",
      "family": "fun_asr_nano",
      "path": "models/Fun-ASR-Nano-2512-GGUF/fun-asr-nano-2512-q8_0.gguf",
      "task": "asr",
      "mode": "offline"
    }
  ]
}
```

```bash
audiocpp_server --config server.json

curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -F model=fun-asr-nano \
  -F language=auto \
  -F file=@speech.wav
```

The endpoint follows the OpenAI multipart transcription shape and returns
`{"text":"...","timing":{...}}`.

## GGUF Conversion

```bash
audiocpp_gguf \
  --input /path/to/Fun-ASR-Nano-2512-hf/model.safetensors \
  --root /path/to/Fun-ASR-Nano-2512-hf \
  --output fun-asr-nano-2512-q8_0.gguf \
  --type q8_0 \
  --family fun_asr_nano \
  --model-spec model_specs/fun_asr_nano.json
```

The standalone output embeds the config, processor config, tokenizer, chat
template, and model package specification. Inspect it with:

```bash
audiocpp_gguf --inspect fun-asr-nano-2512-q8_0.gguf
```

## Validation

The published F16 and Q8_0 files were checked with the same 14.07-second
reference WAV on CPU and NVIDIA H100 CUDA. CLI and OpenAI-compatible server
transcripts matched the official safetensors path. The Q8_0 server validation
completed at approximately 0.0725 RTF on H100.

## License

The checkpoint and converted weights are governed by the FunASR Model Open
Source License Agreement v1.1 distributed with the official model. Review that
agreement before use or redistribution.
