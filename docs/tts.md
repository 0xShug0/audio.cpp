# TTS Models

This page covers speech TTS-style families that do not have a dedicated model page. Qwen3, VeVo2, Seed-VC, ACE-Step, and Stable Audio have dedicated pages.

Common CLI shape:

```bash
audiocpp_cli --task <task> --family <family> --model <model-dir> --backend cuda ...
```

Common options:

| Option | Meaning |
|---|---|
| `--text` | Text, prompt, lyrics, or multi-speaker script, depending on the model. |
| `--voice-ref` | Reference voice WAV for models that support cloning. |
| `--reference-text` | Transcript or prompt text for models that use explicit reference transcripts. |
| `--voice-id` | Built-in voice id for models with packaged voices. |
| `--language` | Model language code when the model requires one. |
| `--text-chunk-size` | Long-form chunk budget in characters. Each model has its own default. |
| `--seed` | Optional fixed seed. If omitted, models that sample use a random seed unless their upstream default is fixed. |

## Chatterbox

Chatterbox is a voice-clone TTS model with an audio-to-audio voice-conversion path. The upstream Chatterbox family also documents paralinguistic tag tokens in newer variants, but the current audio.cpp integration exposes voice cloning and voice conversion rather than a separate tag-control interface.

| Field | Value |
|---|---|
| Family | `chatterbox` |
| Model directory | `models/chatterbox` |
| Tasks | `clon`, `vc` |
| Modes | `offline` |
| Languages | `ar`, `da`, `de`, `el`, `en`, `es`, `fi`, `fr`, `hi`, `it`, `ko`, `ms`, `nl`, `no`, `pl`, `pt`, `sv`, `sw`, `tr` |
| Voice input | Required reference WAV through `--voice-ref`; VC also requires source audio through `--audio` |
| Built-in voices | Not exposed by this integration |

Voice clone:

```bash
audiocpp_cli --task clon --family chatterbox --model models/chatterbox --backend cuda --text "Hello from Chatterbox." --voice-ref assets/resources/b.wav --out out.wav
```

Voice conversion:

```bash
audiocpp_cli --task vc --family chatterbox --model models/chatterbox --backend cuda --audio assets/resources/a.wav --voice-ref assets/resources/b.wav --out converted.wav
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--audio` | WAV path | required for `vc` | Source speech for voice conversion. |
| `--voice-ref` | WAV path | required | Reference speaker audio for cloning, or target speaker audio for voice conversion. |
| `--language` | language code | `en` | Text language. |
| `--text-chunk-size` | integer chars | `128` | Long-form chunk size. |
| `--guidance-scale` | float | `0.5` | CFG strength. |
| `--temperature` | float | `0.8` | T3 sampling temperature. |
| `--top-p` | float | `0.8` | T3 nucleus sampling limit. |
| `--repetition-penalty` | float | `2.0` | T3 repetition penalty. |
| `--max-tokens` | integer | `1000` | Maximum generated T3 tokens per chunk. |
| `--do-sample` | `true`, `false` | `true` | Enable stochastic T3 sampling. |

## Kokoro

Kokoro is a small preset-voice TTS model. Upstream Kokoro supports packaged voice tensors; audio.cpp exposes the packaged voices by id through `--voice-id`.

| Field | Value |
|---|---|
| Family | `kokoro_tts` |
| Model directory | `models/kokoro-82m-v1_0-ggml` |
| Task | `tts` |
| Modes | `offline` |
| Languages | `a` for American English, `b` for British English |
| Voice input | Built-in voice id |
| External voice tensor | Not exposed by the CLI |

```bash
audiocpp_cli --task tts --family kokoro_tts --model models/kokoro-82m-v1_0-ggml --backend cuda --language a --text "Hello from Kokoro." --voice-id af_heart --out out.wav
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--voice-id` | packaged Kokoro voice id | required | Built-in voice tensor name. |
| `--language` | `a`, `b` | `a` | Kokoro language/accent code. |
| `--text-chunk-size` | integer chars | `240` | Long-form chunk size. |

## MioTTS

MioTTS is a 1.7B voice-clone TTS path that uses MioCodec for acoustic decoding. It requires a reference voice.

| Field | Value |
|---|---|
| Family | `miotts` |
| Model directory | `models/MioTTS-1.7B` |
| Task | `tts` |
| Modes | `offline` |
| Languages | Model auto-handles supported text languages; no explicit language selector is exposed |
| Voice input | Required reference WAV through `--voice-ref` |
| Built-in voices | Not exposed |

```bash
audiocpp_cli --task tts --family miotts --model models/MioTTS-1.7B --backend cuda --text "Hello from MioTTS." --voice-ref assets/resources/b.wav --out out.wav
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--voice-ref` | WAV path | required | Reference speaker audio. |
| `--text-chunk-size` | integer chars | `180` | Long-form chunk size. |
| `--max-tokens` | integer | `700` | Maximum generated LM tokens per chunk. |
| `--temperature` | float | `0.8` | LM sampling temperature. |
| `--top-k` | integer | `50` | LM top-k sampling limit. |
| `--top-p` | float | `1.0` | LM nucleus sampling limit. |
| `--repetition-penalty` | float | `1.0` | LM repetition penalty. |
| `--do-sample` | `true`, `false` | `true` | Enable stochastic LM sampling. |
| `--request-option best_of_n_enabled=true|false` | bool | `false` | Run best-of-N candidate selection. |

## MOSS-TTS-Local

MOSS-TTS-Local is the larger local-transformer MOSS TTS path. It supports text-only speech and optional zero-shot voice cloning through the framework speaker-reference interface.

| Field | Value |
|---|---|
| Family | `moss_tts_local` |
| Model directory | `models/MOSS-TTS-Local-Transformer-v1.5` |
| Required codec layout | `audio_tokenizer/` directory inside the model root |
| Task | `tts`, `clon` |
| Modes | `offline` |
| Languages | Model auto-handles supported languages; `--language` can pass a language hint |
| Voice input | Optional reference WAV through `--voice-ref`; transcript through `--reference-text` when known |
| Built-in voices | Not exposed |

Text-only speech:

```bash
audiocpp_cli --task tts --family moss_tts_local --model /path/to/MOSS-TTS-Local-Transformer-v1.5 --backend cuda --text "Hello from MOSS-TTS-Local." --out out.wav
```

Voice clone:

```bash
audiocpp_cli --task clon --family moss_tts_local --model /path/to/MOSS-TTS-Local-Transformer-v1.5 --backend cuda --text "Hello from MOSS-TTS-Local." --voice-ref /path/to/reference.wav --reference-text "Reference transcript when available." --out out.wav
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--voice-ref` | WAV path | not set | Reference speaker audio for cloning. |
| `--reference-text` | text | empty string | Transcript for reference audio. |
| `--language` | language hint | auto | Optional language hint for the prompt template. |
| `--max-tokens` | integer | `4096` | Maximum generated audio frames. |
| `--do-sample` | `true`, `false` | `true` | Enable stochastic audio-token sampling. |
| `--temperature` | float | `1.7` | Audio-token sampling temperature. |
| `--top-p` | float | `0.8` | Audio-token nucleus sampling limit. |
| `--top-k` | integer | `25` | Audio-token top-k sampling limit. |
| `--repetition-penalty` | float | `1.0` | Audio-token repetition penalty. |
| `--request-option text_temperature=<float>` | float | `1.0` | Text-gate sampling temperature. |
| `--request-option text_top_p=<float>` | float | `1.0` | Text-gate nucleus sampling limit. |
| `--request-option text_top_k=<n>` | integer | `50` | Text-gate top-k sampling limit. |
| `--text-chunk-size` | characters | `2048` | Framework long-form text chunk size. |
| `--text-chunk-mode` | `default`, `tag_aware`, `japanese`, `endline` | `default` | Framework long-form text chunking mode. |
| `--session-option moss_tts_local.weight_type=auto|native|f32|f16|bf16|q8_0` | enum | `auto` | Backbone weight storage type. |
| `--session-option moss_tts_local.reference_cache_slots=<n>` | integer slots | `1` | Prepared reference-voice cache slots; set `0` to disable reuse. |

## MOSS-TTS-Nano

MOSS-TTS-Nano is the smaller MOSS TTS path. It supports text-only continuation generation and voice cloning through the framework speaker-reference interface.

| Field | Value |
|---|---|
| Family | `moss_tts_nano` |
| Model directory | `models/MOSS-TTS-Nano-100M` |
| Required codec layout | `audio_tokenizer/` directory inside the model root |
| Task | `tts`, `clon` |
| Modes | `offline` |
| Languages | Model auto-handles supported languages |
| Voice input | Optional reference WAV through `--voice-ref` |
| Built-in voices | Not exposed |

Text-only continuation:

```bash
audiocpp_cli --task tts --family moss_tts_nano --model /path/to/MOSS-TTS-Nano-100M --backend cuda --text "Hello from MOSS-TTS-Nano." --out out.wav
```

Voice clone:

```bash
audiocpp_cli --task clon --family moss_tts_nano --model /path/to/MOSS-TTS-Nano-100M --backend cuda --text "Hello from MOSS-TTS-Nano." --voice-ref /path/to/reference.wav --reference-text "Reference transcript when available." --out out.wav
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--voice-ref` | WAV path | not set | Reference speaker audio for cloning. When omitted, Nano uses text-only continuation mode. |
| `--reference-text` | text | empty string | Transcript for reference audio; valid only with `--voice-ref`. |
| `--max-tokens` | integer | `300` | Maximum generated audio frames per chunk. |
| `--do-sample` | `true`, `false` | `true` | Enable stochastic audio-token sampling. |
| `--temperature` | float | `1.7` | Audio-token sampling temperature. |
| `--top-p` | float | `0.8` | Audio-token nucleus sampling limit. |
| `--top-k` | integer | `25` | Audio-token top-k sampling limit. |
| `--repetition-penalty` | float | `1.0` | Audio-token repetition penalty. |
| `--request-option text_temperature=<float>` | float | `1.5` | Text-gate sampling temperature. |
| `--request-option text_top_p=<float>` | float | `1.0` | Text-gate nucleus sampling limit. |
| `--request-option text_top_k=<n>` | integer | `50` | Text-gate top-k sampling limit. |
| `--text-chunk-size` | characters | `256` | Framework long-form text chunk size. |
| `--text-chunk-mode` | `default`, `tag_aware`, `japanese`, `endline` | `default` | Framework long-form text chunking mode. |
| `--session-option moss_tts_nano.weight_type=native|f32|f16|bf16|q8_0` | enum | `native` | Global and local-frame weight storage type. |
| `--session-option moss_tts_nano.global_weight_type=native|f32|f16|bf16|q8_0` | enum | `native` | Global transformer weight storage type. |
| `--session-option moss_tts_nano.local_frame_weight_type=native|f32|f16|bf16|q8_0` | enum | `native` | Local frame decoder weight storage type. |
| `--session-option moss_tts_nano.global_prefill_graph_arena_mb=<n>` | MB | `256` | Global prefill graph arena size. |
| `--session-option moss_tts_nano.global_decode_graph_arena_mb=<n>` | MB | `128` | Global decode graph arena size. |
| `--session-option moss_tts_nano.global_weight_context_mb=<n>` | MB | `512` | Global transformer weight context size. |
| `--session-option moss_tts_nano.local_frame_graph_arena_mb=<n>` | MB | `64` | Local frame decoder graph arena size. |
| `--session-option moss_tts_nano.local_frame_weight_context_mb=<n>` | MB | `128` | Local frame decoder weight context size. |
| `--session-option moss_tts_nano.audio_tokenizer_encoder_graph_arena_mb=<n>` | MB | `64` | Audio tokenizer encoder graph arena size. |
| `--session-option moss_tts_nano.audio_tokenizer_decoder_graph_arena_mb=<n>` | MB | `64` | Audio tokenizer decoder graph arena size. |
| `--session-option moss_tts_nano.audio_tokenizer_weight_context_mb=<n>` | MB | `128` | Audio tokenizer weight context size. |

## OmniVoice

OmniVoice supports multilingual TTS, voice cloning, voice design, and non-verbal tag tokens. The integration exposes both reference-audio cloning and instruction-based voice design.

| Field | Value |
|---|---|
| Family | `omnivoice` |
| Model directory | `models/OmniVoice` |
| Task | `tts` |
| Modes | `offline` |
| Languages | 600+ languages handled by the model |
| Voice input | `--voice-ref` plus optional `--reference-text`, or instruction text through `--instruct` |
| Built-in voices | Auto voice is supported by the model; CLI examples use clone or design for repeatability |

Voice clone:

```bash
audiocpp_cli --task tts --family omnivoice --model models/OmniVoice --backend cuda --text "Hello from OmniVoice." --voice-ref assets/resources/b.wav --reference-text "Some call me nature. Others call me Mother Nature. I've been here for over 4.5 billion years. 22,500 times longer than you." --out out.wav
```

Voice design:

```bash
audiocpp_cli --task tts --family omnivoice --model models/OmniVoice --backend cuda --text "Hello from OmniVoice." --instruct "female, young adult, moderate pitch" --out out.wav
```

Non-verbal tags are written directly in `--text`. Supported tag spellings include `[laughter]`, `[sigh]`, `[confirmation-en]`, `[question-en]`, `[question-ah]`, `[question-oh]`, `[question-ei]`, `[question-yi]`, `[surprise-ah]`, `[surprise-oh]`, `[surprise-wa]`, `[surprise-yo]`, and `[dissatisfaction-hnn]`.

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--voice-ref` | WAV path | not set | Reference speaker audio for cloning. |
| `--reference-text` | text | empty string | Transcript for reference audio. |
| `--instruct` | text | empty string | Voice-design instruction. |
| `--text-chunk-size` | integer chars | disabled | Optional framework text chunking. |
| `--text-chunk-mode` | `default`, `tag_aware`, `japanese`, `endline` | `tag_aware` | Framework text chunking mode used only when `--text-chunk-size` is set. |
| `--num-inference-steps` | integer | `32` | Decoder diffusion steps. |
| `--guidance-scale` | float | `2.0` | Decoder CFG strength. |
| `--request-option speed=<float>` | float | `1.0` | Speech speed multiplier. |
| `--request-option audio_chunk_duration_seconds=<float>` | seconds | `15.0` | Audio chunk duration used by the model prompt path. |
| `--request-option audio_chunk_threshold_seconds=<float>` | seconds | `30.0` | Audio length threshold before model-side chunking. |
| `--session-option omnivoice.mem_saver=true|false` | bool | `false` | Release staged generator and audio-tokenizer runtime graphs after request phases to reduce resident VRAM. Later requests may rebuild released graphs. |

When `--text-chunk-size` is not set, long OmniVoice requests keep the model-specific automatic punctuation chunker controlled by `audio_chunk_duration_seconds` and `audio_chunk_threshold_seconds`.

## PocketTTS

PocketTTS supports built-in voices and voice cloning. The upstream project also supports exported voice states for fast reuse; the CLI surface here exposes built-in voice ids and reference WAVs.

| Field | Value |
|---|---|
| Family | `pocket_tts` |
| Model directory | `models/pocket-tts` |
| Task | `tts` |
| Modes | `offline` |
| Languages | `english`, `german`, `italian`, `portuguese`, `spanish`|
| Voice input | Built-in voice id or reference WAV |
| Built-in voices | Voice ids depend on the downloaded language package; `alba` is used by the examples |

Preset voice:

```bash
audiocpp_cli --task tts --family pocket_tts --model models/pocket-tts --backend cuda --text "Hello from PocketTTS." --voice-id alba --out out.wav
```

Voice clone:

```bash
audiocpp_cli --task tts --family pocket_tts --model models/pocket-tts --backend cuda --text "Hello from PocketTTS." --voice-ref assets/resources/b.wav --out out.wav
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--load-option language=<name>` | language package name | `english` | Select PocketTTS language package at load time. |
| `--voice-id` | packaged voice id | not set | Built-in voice id. |
| `--voice-ref` | WAV path | not set | Reference speaker audio for cloning. |
| `--text-chunk-size` | integer chars | `256` | Long-form chunk size. |
| `--session-option pocket_tts.voice_state_cache_slots=<n>` | integer slots | `4` | Prepared voice-state cache slots; set `0` to disable reuse. |

## VoxCPM2

VoxCPM2 supports plain TTS, voice design, controllable voice cloning, and an ultimate-clone style that uses both prompt audio and transcript. The CLI expresses voice design with the same text convention as the upstream examples: put the voice/style description in parentheses at the start of `--text`.

| Field | Value |
|---|---|
| Family | `voxcpm2` |
| Model directory | `models/VoxCPM2` |
| Task | `tts` |
| Modes | `offline`, `streaming` |
| Languages | Model auto-handles supported languages |
| Voice input | Optional reference WAV; optional transcript through `--reference-text` |
| Built-in voices | Not exposed |

Voice design:

```bash
audiocpp_cli --task tts --family voxcpm2 --model models/VoxCPM2 --backend cuda --text "(A young woman, gentle and clear voice)Hello from VoxCPM2." --out out.wav
```

Voice clone:

```bash
audiocpp_cli --task tts --family voxcpm2 --model models/VoxCPM2 --backend cuda --text "Hello from VoxCPM2." --voice-ref assets/resources/b.wav --out out.wav
```

Ultimate clone:

```bash
audiocpp_cli --task tts --family voxcpm2 --model models/VoxCPM2 --backend cuda --text "Hello from VoxCPM2." --voice-ref assets/resources/b.wav --reference-text "Some call me nature. Others call me Mother Nature. I've been here for over 4.5 billion years. 22,500 times longer than you." --out out.wav
```

Streaming output:

```bash
audiocpp_cli --task tts --family voxcpm2 --model models/VoxCPM2 --backend cuda --mode streaming --text "Hello from VoxCPM2." --out out.wav
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--text "(style)content"` | text | required | Voice design or style control. |
| `--voice-ref` | WAV path | not set | Reference speaker audio. |
| `--reference-text` | text | empty string | Transcript for ultimate-clone style prompting. |
| `--mode` | `offline`, `streaming` | `offline` | Full-output or streaming run mode. |
| `--session-option voxcpm2.mem_saver=true|false` | bool | `false` | Use tighter graph workspaces and release MiniCPM/AudioVAE request graphs after completion to reduce resident VRAM. |
| `--session-option voxcpm2.prompt_cache_slots=<n>` | integer | `1` | Prompt and prompt-audio embedding cache slots. Set to `0` to disable prompt caching. |
| `--text-chunk-size` | integer chars | `2048` | Long-form chunk size. |
| `--text-chunk-mode` | `default`, `tag_aware`, `japanese`, `endline` | `tag_aware` | Long-form chunking mode; keeps style/tag controls attached to chunks by default. |
| `--max-tokens` | integer | `4096` | Maximum generated AR tokens. |
| `--num-inference-steps` | integer | `10` | Flow matching steps. |
| `--guidance-scale` | float | `2.0` | CFG strength. |

## Higgs Audio v3 TTS

Higgs Audio v3 TTS is a voice-clone TTS model. The current integration uses the framework chunker for long text and keeps the reference prompt state in the model session.

| Field | Value |
|---|---|
| Family | `higgs_tts` |
| Model directory | `models/higgs-audio-v3-tts-4b` |
| Task | `tts` |
| Modes | `offline` |
| Languages | Model auto-handles supported languages |
| Voice input | Reference WAV through `--voice-ref`; transcript through `--reference-text` when known |
| Built-in voices | Not exposed |

```bash
audiocpp_cli --task tts --family higgs_tts --model models/higgs-audio-v3-tts-4b --backend cuda --text "Hello from Higgs Audio." --voice-ref assets/resources/b.wav --reference-text "Some call me nature. Others call me Mother Nature. I've been here for over 4.5 billion years. 22,500 times longer than you." --out out.wav
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--voice-ref` | WAV path | required | Reference speaker audio. |
| `--reference-text` | text | empty string | Transcript for reference audio. |
| `--text-chunk-size` | integer chars | `512` | Long-form chunk size. |
| `--max-tokens` | integer | `1024` | Maximum generated AR tokens per chunk. |
| `--temperature` | float | `0.8` | AR sampling temperature. |
| `--top-k` | integer | `30` | AR top-k sampling limit. The narrower default is less prone to premature EOC than the Python client's `50`. |
| `--top-p` | float | `0.8` | AR nucleus sampling limit. The Python client's unfiltered equivalent is `1.0`. |
| `--repetition-penalty` | float | `1.1` | Accepted for Python API compatibility; Higgs audio-code sampling does not consume it. |

## IndexTTS2

IndexTTS2 is a Chinese and English TTS model with voice cloning and expressive emotion controls. It requires a speaker reference through the framework `--voice-ref` path.

| Field | Value |
|---|---|
| Family | `index_tts2` |
| Model directory | `models/IndexTTS-2` |
| Task | `tts`, `clon` |
| Modes | `offline` |
| Languages | `zh`, `en` |
| Voice input | Required reference WAV through `--voice-ref` |
| Built-in voices | Not exposed |

Voice clone:

```bash
audiocpp_cli --task clon --family index_tts2 --model /path/to/IndexTTS-2 --backend cuda --language en --text "Hello from IndexTTS2." --voice-ref /path/to/reference.wav --out out.wav
```

Emotion text:

```bash
audiocpp_cli --task tts --family index_tts2 --model /path/to/IndexTTS-2 --backend cuda --language zh --text "今天的演示会更有情绪。" --voice-ref /path/to/reference.wav --emotion "你吓死我了！你是鬼吗？" --request-option emotion_alpha=0.6 --out out.wav
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--voice-ref` | WAV path | required | Reference speaker audio. |
| `--language` | `zh`, `en` | empty | Text language label. |
| `--emotion` | text | not set | Emotion-text conditioning through the framework style field. |
| `--request-option emotion_alpha=<float>` | float in `[0, 1]` | `1.0` | Blend strength for explicit emotion conditioning. |
| `--request-option emotion_vector=<v0,...,v7>` | 8 floats | not set | Explicit emotion vector. |
| `--request-option use_emotion_text=true|false` | bool | `false` | Infer emotion from text. |
| `--request-option use_random_emotion=true|false` | bool | `false` | Use random emotion weights in the emotion mixer. |
| `--request-option interval_silence_ms=<n>` | milliseconds | `200` | Silence inserted between generated text chunks. |
| `--text-chunk-size` | characters | not set | Optional framework outer text chunk size. When omitted, IndexTTS2 keeps its internal tokenizer segmentation. |
| `--text-chunk-mode` | `default`, `tag_aware`, `japanese`, `endline` | `default` | Framework chunking mode used only when `--text-chunk-size` is set. |
| `--max-tokens` | integer | model default | Maximum generated GPT mel tokens. |
| `--temperature` | float | model default | GPT sampling temperature. |
| `--top-p` | float | model default | GPT nucleus sampling limit. |
| `--top-k` | integer | model default | GPT top-k sampling limit. |
| `--repetition-penalty` | float | model default | GPT repetition penalty. |
| `--do-sample` | `true`, `false` | model default | Enable stochastic GPT sampling. |
| `--session-option index_tts2.mem_saver=true|false` | bool | `false` | Release staged reference and conditioning graphs after request phases. |
| `--session-option index_tts2.weight_type=native|f32|f16|bf16|q8_0` | enum | `native` | Matmul weight storage type. |
| `--session-option index_tts2.conv_weight_type=native|f32|f16` | enum | `native` | Convolution weight storage type. |
| `--session-option index_tts2.speaker_cache_slots=<n>` | integer slots | `1` | Prepared speaker-reference cache slots; set `0` to disable reuse. |
| `--session-option index_tts2.emotion_cache_slots=<n>` | integer slots | `1` | Prepared emotion-reference cache slots; set `0` to disable reuse. |
| `--session-option index_tts2.emotion_text_cache_slots=<n>` | integer slots | `1` | Emotion-text weight cache slots; set `0` to disable reuse. |
| `--session-option index_tts2.gpt_graph_arena_mb=<n>` | MB | model default | GPT graph arena size. |
| `--session-option index_tts2.s2mel_graph_arena_mb=<n>` | MB | model default | S2Mel graph arena size. |
| `--session-option index_tts2.reference_graph_arena_mb=<n>` | MB | model default | Reference encoder and codec graph arena size. |
| `--session-option index_tts2.emotion_text_prefill_graph_arena_mb=<n>` | MB | model default | Emotion-text prefill graph arena size. |
| `--session-option index_tts2.emotion_text_decode_graph_arena_mb=<n>` | MB | model default | Emotion-text cached-step graph arena size. |
| `--session-option index_tts2.emotion_text_max_new_tokens=<n>` | tokens | `256` | Maximum generated tokens for emotion-text classification. |
| `--session-option index_tts2.weight_context_mb=<n>` | MB | model default | Shared weight context size. |

## Irodori-TTS

Irodori-TTS is Japanese TTS. The 500M model supports no-reference and reference-conditioned speech; the 600M VoiceDesign model adds caption-based voice design.

| Field | Value |
|---|---|
| Family | `irodori_tts` |
| Model directories | `models/Irodori-TTS-500M-v3`, `models/Irodori-TTS-600M-v3-VoiceDesign` |
| Required shared tokenizer | `models/llm-jp-3-150m/tokenizer.json` |
| Required shared codec | `models/Semantic-DACVAE-Japanese-32dim/weights.safetensors` |
| Tasks | `tts`, `clon`, `vdes` |
| Modes | `offline` |
| Languages | `ja` |
| Voice input | Optional reference WAV, no-reference mode, or caption for VoiceDesign |
| Built-in voices | Not exposed |

No-reference speech:

```bash
audiocpp_cli --task tts --family irodori_tts --model /path/to/Irodori-TTS-500M-v3 --backend cuda --language ja --text "今日は短い確認です。やさしく、聞き取りやすい声でお願いします。" --request-option no_ref=true --out out.wav
```

Voice design:

```bash
audiocpp_cli --task vdes --family irodori_tts --model /path/to/Irodori-TTS-600M-v3-VoiceDesign --backend cuda --language ja --text "本日はお越しいただき、誠にありがとうございます。" --request-option caption="落ち着いた大人の男性。深く響く声で丁寧に話している。" --request-option no_ref=true --out out.wav
```

Reference-conditioned speech:

```bash
audiocpp_cli --task clon --family irodori_tts --model /path/to/Irodori-TTS-500M-v3 --backend cuda --language ja --text "同じ声で短く話します。" --voice-ref /path/to/reference.wav --request-option no_ref=false --out out.wav
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--language` | `ja` | `ja` | Spoken language. |
| `--request-option no_ref=true|false` | bool | `true` | Use no-reference generation. Set `false` with `--voice-ref` for reference conditioning. |
| `--voice-ref` | WAV path | not set | Optional speaker reference. |
| `--request-option caption=<text>` | text | empty string | Voice-design caption for the 600M model. |
| `--num-inference-steps` | integer | `40` | RF diffusion steps. |
| `--duration-seconds` | seconds | `0` | Force duration when positive; `0` uses model-predicted duration. |
| `--text-chunk-mode` | `japanese`, `endline` | `endline` | Long-form chunking mode; `endline` splits only at sentence punctuation followed by a line break or end of input. |
| `--request-option duration_scale=<float>` | float | `1.0` | Scale predicted duration. |
| `--request-option min_seconds=<float>` | seconds | `0.5` | Minimum generated duration. |
| `--request-option max_seconds=<float>` | seconds | `30` | Maximum generated duration. |
| `--request-option text_guidance_scale=<float>` | float | `3.0` | Text CFG strength. |
| `--request-option speaker_guidance_scale=<float>` | float | `5.0` | Speaker CFG strength. |
| `--request-option caption_guidance_scale=<float>` | float | `3.0` | Caption CFG strength. |
| `--request-option guidance_mode=<name>` | `independent` | `independent` | CFG combination mode. |
| `--request-option trim_tail=true|false` | bool | `true` | Trim trailing silence-like samples. |
| `--session-option irodori_tts.mem_saver=true|false` | bool | `true` | Release staged runtime graphs after request phases to reduce resident VRAM. Set `false` to keep graphs resident for maximum reuse. |
| `--session-option irodori_tts.weight_type=native|f32|f16|bf16|q8_0` | enum | `native` | Model weight storage type. |
| `--session-option irodori_tts.codec_weight_type=native|f32|f16|q8_0` | enum | `native` | DACVAE codec weight storage type. |
| `--session-option irodori_tts.reference_cache_slots=<n>` | integer slots | `1` | Prepared reference-speaker cache slots; set `0` to disable reuse. |
| `--session-option irodori_tts.condition_graph_arena_mb=<n>` | MB | `256` | Condition encoder graph arena size. |
| `--session-option irodori_tts.rf_graph_arena_mb=<n>` | MB | `768` | RF sampler graph arena size. |
| `--session-option irodori_tts.codec_graph_arena_mb=<n>` | MB | `512` | DACVAE codec graph arena size. |
| `--session-option irodori_tts.condition_weight_context_mb=<n>` | MB | `512` | Condition encoder weight context size. |
| `--session-option irodori_tts.rf_weight_context_mb=<n>` | MB | `768` | RF sampler weight context size. |
| `--session-option irodori_tts.codec_weight_context_mb=<n>` | MB | `512` | DACVAE codec weight context size. |

## Supertonic

Supertonic 3 is a preset-voice multilingual TTS model. It does not use external speaker references in the current integration.

| Field | Value |
|---|---|
| Family | `supertonic` |
| Model directory | `models/supertonic-3` |
| Task | `tts` |
| Modes | `offline`, `streaming` |
| Languages | `en`, `ko`, `ja`, `ar`, `bg`, `cs`, `da`, `de`, `el`, `es`, `et`, `fi`, `fr`, `hi`, `hr`, `hu`, `id`, `it`, `lt`, `lv`, `nl`, `pl`, `pt`, `ro`, `ru`, `sk`, `sl`, `sv`, `tr`, `uk`, `vi`, `na` |
| Voice input | Built-in preset voice id |
| Built-in voices | `M1`-`M5`, `F1`-`F5` |

```bash
audiocpp_cli --task tts --family supertonic --model /path/to/supertonic-3 --backend cuda --language en --text "Hello from Supertonic." --voice-id M1 --out out.wav
```

Streaming output:

```bash
audiocpp_cli --task tts --family supertonic --model /path/to/supertonic-3 --backend cuda --mode streaming --language en --text "Hello from Supertonic." --voice-id M1 --out out.wav
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--voice-id` | `M1`-`M5`, `F1`-`F5` | `M1` | Preset voice. |
| `--language` | language code | `en` | Text language. |
| `--num-inference-steps` | integer | `8` | Flow denoising steps. |
| `--request-option speaking_rate=<float>` | float | `1.05` | Speech speed multiplier. |
| `--seed` | integer | `1234` | Noise seed. |
| `--text-chunk-size` | characters | `300`, or `120` for `ko`/`ja` | Framework long-form text chunk size. |
| `--text-chunk-mode` | `default`, `tag_aware`, `japanese`, `endline` | `default` | Framework long-form text chunking mode. |
| `--session-option supertonic.weight_type=native|f32|f16|bf16|q8_0` | enum | `native` | Weight storage type. |
| `--session-option supertonic.style_cache_slots=<n>` | integer slots | `4` | Preset voice style cache slots; set `0` to disable reuse. |

## VibeVoice

VibeVoice is a long-form multi-speaker TTS model, available in 1.5B and 7B sizes. Prompts use speaker-labeled lines, and speaker reference WAVs are provided in the same order as the speaker ids.

| Field | Value |
|---|---|
| Family | `vibevoice` |
| Model directory | `models/VibeVoice-1.5B` (or `models/VibeVoice-7B`) |
| Task | `tts` |
| Modes | `offline` |
| Languages | Model auto-handles supported languages |
| Voice input | Up to four speaker reference WAVs through `voice_samples` |
| Text format | Lines like `Speaker 1: ... Speaker 2: ...`; ids are normalized internally |
| Long-form | No text chunking; generation uses the model long-form path |
| LoRA | Optional PEFT decoder adapter through `--load-option vibevoice.lora` |

Both sizes share the same CLI surface and the same Qwen2.5 tokenizer; the 7B is simply larger (hidden size 3584 vs 1536) and needs a matching 7B LoRA if one is used.

```bash
audiocpp_cli --task tts --family vibevoice --model models/VibeVoice-1.5B --backend cuda --text "Speaker 1: Hello. Speaker 2: Nice to meet you." --request-option voice_samples=assets/resources/a.wav,assets/resources/b.wav --out out.wav
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--request-option voice_samples=a.wav,b.wav` | comma-separated WAVs | not set | Speaker reference WAVs, ordered by speaker id. |
| `--guidance-scale` | float | `1.3` | Classifier-free guidance scale. |
| `--num-inference-steps` | integer | `10` | Diffusion steps per audio chunk. |
| `--max-tokens` | integer, `0` for unlimited | `0` | Maximum generated decoder tokens. |
| `--request-option max_length_times=<float>` | float | `2.0` | Generation length multiplier. |
| `--do-sample` | `true`, `false` | `false` | Enable stochastic decoder sampling. |
| `--temperature` | float | `1.0` | Decoder sampling temperature. |
| `--top-k` | integer | `50` | Decoder top-k sampling limit. |
| `--top-p` | float | `1.0` | Decoder nucleus sampling limit. |
| `--load-option vibevoice.lora=<path>` | fine-tune adapter dir | not set | Overlay a fine-tune at load time: the language-model LoRA is delta-merged into the decoder linears, and the diffusion head and acoustic/semantic connectors (when present in the adapter dir) replace their base tensors. Dims must match the base model size. |
| `--load-option vibevoice.lora_scale=<float>` | float | `lora_alpha / r` | Override the LoRA merge scale from `adapter_config.json`. |

The adapter follows the PEFT training layout: `adapter_model.safetensors` + `adapter_config.json` for the language-model LoRA, plus optional `diffusion_head/model.safetensors` (or `diffusion_head_full.bin`), `acoustic_connector/pytorch_model.bin`, and `semantic_connector/pytorch_model.bin` for the fully fine-tuned components. Everything is applied at load time, so it composes with the `vibevoice.*_weight_type` quantization options and adds no per-step cost; the overlay is logged with `--log`. Use a 1.5B adapter with `VibeVoice-1.5B` and a 7B adapter with `VibeVoice-7B`; a size mismatch is rejected with a descriptive error. The same option may instead be passed as `--session-option vibevoice.lora` (but not via both at once).

For backend weight-type controls, use `audiocpp_cli --inspect --model <model-dir> --family <family>`.
