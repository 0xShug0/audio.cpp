# OuteTTS

OuteTTS 1.0 1B generates 24 kHz speech with a Llama text/audio-token model and the IBM DAC 1.5 kbps codec. The integration supports both no-reference generation and native voice cloning. For cloning, the DAC encoder turns a reference WAV into the two codec-token streams used to condition the language model; no separate encoder model or speaker-profile file is required.

| Field | Value |
|---|---|
| Family | `outetts` |
| Model directory | `models/Llama-OuteTTS-1.0-1B` |
| Task | `tts`, `clon` |
| Modes | `offline` |
| Languages | `en`, `ar`, `zh`, `nl`, `fr`, `de`, `it`, `ja`, `ko`, `lt`, `ru`, `es`, plus moderate-data `pt`, `be`, `bn`, `ka`, `hu`, `lv`, `fa`, `pl`, `sw`, `ta`, `uk` |
| Voice input | Optional reference WAV plus its transcript |
| Output | mono 24 kHz WAV |

Install both the language model and its DAC dependency:

```bash
python tools/model_manager.py install outetts_1_0_1b --models-dir models
```

The standalone Q8 GGUF package can also be downloaded directly from
[mirek190/audio.cpp](https://huggingface.co/mirek190/audio.cpp/tree/main/Text%20to%20audio%20(TTS)).
Use `Llama-OuteTTS-1.0-1B_Q8.gguf` as the `--model` path when running the
GGUF examples below.

Run the safetensors package:

```bash
audiocpp_cli --task tts --family outetts \
  --model models/Llama-OuteTTS-1.0-1B \
  --backend cuda --text "Hello from OuteTTS." \
  --max-tokens 1024 --out out.wav
```

Clone a voice with either the safetensors package or standalone GGUF. The reference transcript must match the spoken reference audio. Around ten seconds of clean speech is recommended; the maximum accepted reference length is twenty seconds. The standalone GGUF described below contains Qwen3 Forced Aligner and activates it automatically for accurate per-word codec conditioning:

```bash
audiocpp_cli --task clon --family outetts \
  --model models/Llama-OuteTTS-1.0-1B-Q8_0/model.gguf \
  --backend cuda \
  --voice-ref reference.wav \
  --reference-text "The exact words spoken in reference.wav." \
  --request-option reference_language=en \
  --text "This sentence uses the cloned voice." \
  --max-tokens 1024 --out cloned.wav
```

`--task tts` with the same `--voice-ref` and `--reference-text` options also enables speaker conditioning, which is useful for clients that expose one TTS route. Safetensors packages and older OuteTTS GGUFs do not contain the aligner. For those models, pass `outetts.aligner_model_path`; cloning fails clearly instead of using unreliable estimated word boundaries.

The installer places `DAC.speech.v1.0` and `Qwen3-ForcedAligner-0.6B` beside the OuteTTS directory. It converts the official DAC checkpoint to a safe tensor source. To do that conversion manually:

```bash
python tools/convert_outetts_dac.py \
  models/DAC.speech.v1.0/weights_24khz_1.5kbps_v1.0.pth \
  models/DAC.speech.v1.0/model.safetensors
```

Pack the language model, DAC, Qwen3 Forced Aligner, and sidecars into one standalone Q8 GGUF:

```bash
audiocpp_gguf \
  --input model_weights=models/Llama-OuteTTS-1.0-1B/model.safetensors \
  --input dac_weights=models/DAC.speech.v1.0/model.safetensors \
  --input aligner_weights=models/Qwen3-ForcedAligner-0.6B/model.safetensors \
  --root models/Llama-OuteTTS-1.0-1B \
  --sidecar models/DAC.speech.v1.0/config.json=dac/config.json \
  --sidecar models/Qwen3-ForcedAligner-0.6B/config.json=aligner/config.json \
  --sidecar models/Qwen3-ForcedAligner-0.6B/generation_config.json=aligner/generation_config.json \
  --sidecar models/Qwen3-ForcedAligner-0.6B/preprocessor_config.json=aligner/preprocessor_config.json \
  --sidecar models/Qwen3-ForcedAligner-0.6B/tokenizer_config.json=aligner/tokenizer_config.json \
  --sidecar models/Qwen3-ForcedAligner-0.6B/vocab.json=aligner/vocab.json \
  --sidecar models/Qwen3-ForcedAligner-0.6B/merges.txt=aligner/merges.txt \
  --output models/Llama-OuteTTS-1.0-1B-Q8_0/model.gguf \
  --type q8_0
```

The resulting GGUF contains all three tensor groups, the model specification, and all required sidecars. It does not need the original directories. Normal TTS does not run the embedded aligner; it is initialized only when reference cloning is requested:

```bash
audiocpp_cli --task tts --family outetts \
  --model models/Llama-OuteTTS-1.0-1B-Q8_0/model.gguf \
  --backend cuda --text "Hello from the standalone GGUF." \
  --max-tokens 1024 --out out.wav
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--max-tokens` | integer | automatic | Maximum audio-token generation length per chunk. When omitted, OuteTTS estimates a safe budget from the chunk's word and character counts. An explicit smaller cap causes additional text splitting; a chunk that unexpectedly reaches the cap is retried as smaller chunks instead of silently truncating speech. |
| `--temperature` | float | `0.4` for cloning; model default for TTS | Sampling temperature. Voice cloning follows the official OuteTTS default without changing temperature between words. |
| `--top-k` | integer | `40` | Top-k sampling limit. |
| `--top-p` | float | `0.9` | Nucleus sampling limit. |
| `--request-option min_p=<float>` | float | `0.05` | Minimum probability relative to the most likely token. |
| `--repetition-penalty` | float | `1.1` | Repetition penalty over the latest 64 tokens. |
| `--request-option seed=<n>` | integer | native clone: `4099`; quantized clone: `42` | Deterministic sampling seed. The defaults were separately verified for the native and Q8 cloning paths. |
| `--text-chunk-size` | characters | `256` | Initial framework long-form text chunk size. Each chunk is split further when needed to fit `max_tokens`, generated and decoded in the same loaded session, then appended to the output WAV. |
| `--text-chunk-mode` | `default`, `tag_aware`, `japanese`, `endline` | `default` | Framework long-form text chunking mode. |
| `--reference-text` | text | none | Exact transcript of `--voice-ref`; required for voice cloning. |
| `--request-option reference_language=<code>` | language code | `en` | Language used by the optional reference aligner. |
| `--session-option outetts.weight_type=native|f32|f16|bf16|q8_0` | enum | `native` | Language-model weight storage type. For CUDA voice cloning, quantized weights remain compact in the GGUF but are expanded to F32 in VRAM to avoid generation divergence over long reference-codec prompts. Normal TTS and CPU cloning keep the selected type. |
| `--session-option outetts.aligner_model_path=<path>` | model path | embedded aligner | Optional external Qwen3 Forced Aligner override, required only for safetensors packages and older GGUFs without the embedded aligner. |
| `--session-option outetts.reference_cache_slots=<n>` | integer | `1` | LRU slots for prepared reference profiles (alignment, DAC codes, and word features). Set `0` to disable reuse. |
| `--session-option outetts.mem_saver=true|false` | bool | `false` | Release the reusable Llama cached-step graph after each generated chunk and release the aligner runtime after preparing a reference. Model and DAC weights stay resident; later requests rebuild released state. |

With logging enabled, OuteTTS reports framework chunk count and token budget, per-chunk word/character counts, recommended and effective generation limits, the natural stop reason, reference-profile cache hits/evictions, Llama runtime and step-graph rebuild/reuse, released cache capacity, and timings for reference alignment, DAC encode/decode, prompt construction, generation, and the complete session request. See [OuteTTS validation](../reports/outetts_validation.md) for the reproducible long-lived session and memory test.
