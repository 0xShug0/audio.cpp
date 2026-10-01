# Index-Echo-S2TT

[Index-Echo-S2TT](https://huggingface.co/IndexTeam/Index-Echo-S2TT-2B)
translates Chinese speech into English, Spanish, or Japanese text. It returns
the source transcript and translation as timestamped subtitle cues.

| Field | Value |
|---|---|
| Family | `index_echo` |
| Task | `asr` |
| Mode | `offline` |
| Input | Chinese speech |
| Output | Timestamp, source transcript, and translation per sentence |
| Checkpoints | 2B: original precision, Q8_0, experimental Q4_K; 9B: original precision and Q8_0 |

## Convert

Download the [official 2B checkpoint](https://huggingface.co/IndexTeam/Index-Echo-S2TT-2B),
then create the GGUF with the debug converter and the model-specific conversion
script. The resulting file includes the model configuration and tokenizer.

```bash
conda run -n qwen3-tts python tests/index_echo/convert_s2tt_gguf.py \
  --source-dir /path/to/Index-Echo-S2TT-2B \
  --output-dir /path/to/Index-Echo-S2TT-GGUF
```

To convert the [official 9B checkpoint](https://huggingface.co/IndexTeam/Index-Echo-S2TT-9B),
pass its source directory and a storage type:

```bash
conda run -n qwen3-tts python tests/index_echo/convert_s2tt_gguf.py \
  --source-dir /path/to/Index-Echo-S2TT-9B \
  --output-dir /path/to/Index-Echo-S2TT-GGUF --type q8_0
```

The same `--type` choices apply to 2B. Use `orig` for original precision,
`q8_0` for 8-bit weights, or `q4_k` for the smallest 2B package. With native
BF16 decoder weights, `orig` also rounds decoder activations to BF16, as in
the official Python default. The 2B Q4_K package keeps its linear-attention
QKV projections at Q8_0 because GGML Q4_K
cannot quantize those tensors. **9B Q4_K is not packaged:** it repeated or
omitted speech on short clips and returned incomplete subtitle cues on a
long-form request. Use 9B Q8_0 or original precision instead.

## CLI

```bash
build/debug/bin/audiocpp_cli --task asr --family index_echo \
  --model /path/to/Index-Echo-S2TT-GGUF/index-echo-s2tt-2b-orig.gguf \
  --backend cuda --audio speech.wav \
  --request-option target_language=en \
  --text-out subtitles.txt --log
```

Select `index-echo-s2tt-9b-q8_0.gguf` for the 9B variant.

For long recordings, the runtime processes windows up to 60 seconds and carries
the preceding five windows' transcript and translation as context. The `auto`
mode chooses quiet boundaries. It does not run the official Python Silero VAD;
window boundaries and, consequently, some output may differ.

## Request Options (use with `--request-option`)

| Option | Default | Meaning |
|---|---|---|
| `target_language` | `en` | `en`, `es`, or `ja`. |
| `max_text_tokens` | `2000` | Maximum generated text tokens per window. |
| `temperature` | `0` | Zero uses greedy decoding; a positive value samples and can interrupt repetition. |
| `seed` | `0` | Sampling seed when temperature is positive. |
| `glossary` | empty | Translation terminology supplied in the prompt. |
| `audio_chunk_mode` | `auto` | `auto` or `quiet_energy` uses quiet boundaries; `fixed` uses equal windows; `none` processes the full input as one window. |
| `audio_chunk_duration_sec` | `60` | Maximum window duration in seconds, from 1 to 60. |

## Session Options (use with `--session-option`)

| Option | Default | Meaning |
|---|---|---|
| `index_echo.weight_type` | `native` | Runtime weight storage type. |

Index-Echo-S2ST speech generation is not supported by this family yet.
