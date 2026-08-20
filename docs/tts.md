# TTS Models

| Model | Family | Task(s) | Quick Start |
|---|---|---|---|
| Qwen3 TTS | `qwen3_tts` | `tts`, `vdes` | [Qwen3 TTS](#qwen3-tts) |
| Chatterbox | `chatterbox` | `clon`, `vc` | [Chatterbox](#chatterbox) |
| Confucius4-TTS | `confucius4_tts` | `clon` | [Confucius4-TTS](#confucius4-tts) |
| DramaBox | `dramabox` | `tts`, `clon` | [DramaBox](#dramabox) |
| DotTTS | `dots_tts` | `tts`, `clon` | [DotTTS](#dottts) |
| MioTTS | `miotts` | `tts` | [MioTTS](#miotts) |
| MOSS-TTS-Local | `moss_tts_local` | `tts`, `clon` | [MOSS-TTS-Local](#moss-tts-local) |
| MOSS-TTS-Nano | `moss_tts_nano` | `tts`, `clon` | [MOSS-TTS-Nano](#moss-tts-nano) |
| MiniMax-H3 | `minimax_h3` | `gen` dialogue audio | [MiniMax-H3](community_models/minimax_h3.md) |
| MagpieTTS | `magpie_tts` | `tts` | [MagpieTTS](#magpietts), [full guide](models/magpie_tts.md) |
| NeuTTS | `neutts` | `tts` | [NeuTTS](#neutts) |
| OmniVoice | `omnivoice` | `tts` | [OmniVoice](#omnivoice), [full guide](models/omnivoice.md) |
| PocketTTS | `pocket_tts` | `tts` | [PocketTTS](#pockettts) |
| VoxCPM2 | `voxcpm2` | `tts`, `vdes` | [VoxCPM2](#voxcpm2) |
| Higgs Audio v3 TTS | `higgs_audio_tts` | `tts` | [Higgs Audio v3 TTS](#higgs-audio-v3-tts) |
| Fish Audio S2 Pro | `fish_audio` | `tts` | [Fish Audio S2 Pro](#fish-audio-s2-pro) |
| IndexTTS2 | `index_tts2` | `tts` | [IndexTTS](models/index_tts.md) |
| IndexTTS2.5 | `index_tts2` (variant `2.5`) | `tts` | [IndexTTS](models/index_tts.md) |
| Irodori-TTS | `irodori_tts` | `tts`, `vdes` | [Irodori-TTS](#irodori-tts) |
| GLM-TTS | `glm_tts` | `tts`, `clon` | [GLM-TTS](#glm-tts) |
| Inflect Micro v2 | `inflect_v2` | `tts` | [Inflect v2](#inflect-v2) |
| OuteTTS | `outetts` | `tts`, `clon` | [OuteTTS](#outetts) |
| Supertonic | `supertonic` | `tts` | [Supertonic](#supertonic) |
| VibeVoice | `vibevoice` | `tts` | [VibeVoice](#vibevoice) |

This page covers speech TTS-style families. MiniMax-H3 appears here for prompt-driven dialogue audio, but it uses the generation route (`--task gen`) rather than the normal speech route (`--task tts`). Detailed route manuals live under `docs/models/` or `docs/community_models/` when a model needs more space.

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

## Qwen3 TTS

Qwen3 TTS supports reference voice cloning, voice design, and packaged custom voices. See [Qwen3 models](models/qwen3.md) for the full Base, VoiceDesign, CustomVoice, ASR, and forced-alignment manual.

```bash
audiocpp_cli --task tts --family qwen3_tts --model models/Qwen3-TTS-12Hz-1.7B-Base --backend cuda --text "Hello from Qwen3 TTS." --voice-ref assets/resources/b.wav --out out.wav
```

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

## Confucius4-TTS

Confucius4-TTS is an experimental multilingual voice-cloning TTS model packaged as a standalone GGUF bundle. It supports offline generation and streaming text input, using reference speech, language-aware text normalization, T2S semantic generation, S2A flow matching, style encoding, semantic audio features, and BigVGAN vocoding.

| Field | Value |
|---|---|
| Family | `confucius4_tts` |
| GGUF model | `models/Confucius4-TTS-GGUF/confucius4-tts-orig.gguf` |
| Task | `clon` |
| Modes | `offline`, `streaming` |
| Languages | `zh`, `en`, `ja`, `ko`, `de`, `fr`, `es`, `id`, `it`, `th`, `pt`, `ru`, `ms`, `vi` |
| Voice input | Required reference WAV through `--voice-ref` |
| Built-in voices | Not exposed |
| Status | Experimental |

Language support note: English (`en`) and Chinese (`zh`) are the currently validated and normalized paths. Other advertised language codes are experimental best-effort cross-language cloning paths; text frontend normalization is incomplete for them, so pronunciation and reading quality may vary.

Voice clone:

```bash
audiocpp_cli --task clon --family confucius4_tts --model models/Confucius4-TTS-GGUF/confucius4-tts-orig.gguf --backend cuda --language en --text "Hello from Confucius four TTS." --voice-ref assets/resources/b.wav --out out.wav
```

Streaming session:

```bash
audiocpp_cli --task clon --family confucius4_tts --model models/Confucius4-TTS-GGUF/confucius4-tts-orig.gguf --backend cuda --mode streaming --language en --text "Hello from the streaming path." --voice-ref assets/resources/b.wav --out out.wav
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--voice-ref` | WAV path | required | Reference speaker audio for cloning. |
| `--language` | language code | `zh` | Target synthesis language code. |
| `--temperature` | float | `0.8` | T2S sampling temperature. |
| `--top-p` | float | `0.8` | T2S nucleus sampling probability. |
| `--top-k` | integer | `30` | T2S top-k sampling limit. |
| `--num-beams` | integer | `3` | T2S beam count; use `1` for single-beam sampling. |
| `--repetition-penalty` | float | `10.0` | T2S repetition penalty. |
| `--max-tokens` | integer | `1520` | Maximum T2S semantic sequence length including prompt tokens. |
| `--num-inference-steps` | integer | `25` | S2A flow-matching step count. |
| `--guidance-scale` | float | `0.7` | S2A classifier-free guidance scale. |
| `--text-chunk-size` | integer tokens | `80` | Maximum text tokens per generated segment. |
| `--text-chunk-mode` | `default`, `tag_aware`, `japanese`, `endline` | `default` | Framework text chunking mode. |
| `--request-option cross_fade_duration_sec=<seconds>` | seconds | `0.3` | Cross-fade duration between generated segments. |
| `--request-option edge_fade_duration_sec=<seconds>` | seconds | `0.1` | Fade duration applied at segment edges. |
| `--request-option edge_pad_duration_sec=<seconds>` | seconds | `0.1` | Silence padding applied at segment edges. |
| `--seed` | integer | `1234` | Seed for T2S sampling and S2A noise initialization. |
| `--session-option confucius4_tts.mem_saver=true\|false` | bool | `false` | Release staged graphs after request phases; default keeps them cached for reuse. |

## DramaBox

DramaBox is an experimental English expressive TTS and voice-cloning model packaged as a standalone GGUF bundle. It combines Gemma text conditioning, diffusion sampling, reference-audio conditioning, long-form chunking, and 48 kHz stereo output.

| Field | Value |
|---|---|
| Family | `dramabox` |
| GGUF model | `models/DramaBox-GGUF/dramabox-q8_0.gguf` |
| Tasks | `tts`, `clon` |
| Modes | `offline` |
| Languages | `en` |
| Voice input | Optional reference WAV through `--voice-ref` |
| Built-in voices | Not exposed |
| Status | Experimental |

Text-only speech:

```bash
audiocpp_cli --task tts --family dramabox --model models/DramaBox-GGUF/dramabox-q8_0.gguf --backend cuda --text "Hello from DramaBox." --out out.wav
```

Voice clone:

```bash
audiocpp_cli --task clon --family dramabox --model models/DramaBox-GGUF/dramabox-q8_0.gguf --backend cuda --text "Hello from DramaBox." --voice-ref assets/resources/b.wav --out out.wav
```

Older prebuilts that reject `--task clon` can use `--task tts --voice-ref ...`; the same reference-conditioning path is used.

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--voice-ref` / `--target-voice` | WAV path | not set | Reference voice for cloning; omitted requests text-only speech. |
| `--request-option negative_prompt=<text>` | string | built-in quality prompt | Negative text conditioning when classifier-free guidance is enabled. |
| `--request-option duration_sec=<seconds>` | seconds | `0` | Explicit target duration; `0` uses prompt-duration estimation. |
| `--num-inference-steps` | integer | `30` | Diffusion sampling steps. |
| `--guidance-scale` | float | `2.5` | Classifier-free guidance scale. Values greater than `1` enable CFG. |
| `--request-option spatio_temporal_guidance_scale=<float>` | float | `1.5` | Spatio-temporal guidance scale. Values greater than `0` enable STG. |
| `--request-option duration_scale=<float>` | float | `1.1` | Multiplier applied to the estimated prompt duration when `duration_sec` is `0`. |
| `--request-option reference_duration_sec=<seconds>` | seconds | `10.0` | Reference voice crop/repeat duration. |
| `--request-option guidance_rescale=auto\|<number>` | string | `auto` | Guidance rescale mode or explicit numeric value. |
| `--request-option audio_chunk_threshold_sec=<seconds>` | seconds | `45.0` | Estimated duration threshold that switches to long-form chunking. |
| `--request-option audio_chunk_duration_sec=<seconds>` | seconds | `37.0` | Target estimated duration for each long-form chunk. |
| `--request-option cross_fade_duration_sec=<seconds>` | seconds | `0.05` | Equal-power cross-fade between long-form chunks. |
| `--seed` | integer | `42` | Torch-compatible CUDA noise seed for diffusion sampling. |
| `--session-option dramabox.perf_mode=off\|flash_attention` | enum | `off` | Attention implementation mode. `off` keeps the exact reference-query attention path; `flash_attention` enables the optimized path. |
| `--session-option dramabox.mem_saver=true\|false` | bool | `false` | Release staged runtime graphs and weights immediately after each request phase to reduce peak and resident VRAM; default keeps components cached for reuse. |

## DotTTS

DotTTS is an experimental multilingual TTS and voice-cloning family with SOAR
and MeanFlow GGUF packages. SOAR is the default download. See
[DotTTS](models/dots_tts.md) for template, streaming, chunking, and full option
details.

```bash
python3 tools/model_manager_v2.py install dots_tts_soar_q8_0

audiocpp_cli --task tts --family dots_tts \
  --model models/DotTTS-SOAR-GGUF/dots-tts-soar-q8_0.gguf \
  --backend cuda \
  --text "Our field team finished the morning inspection and prepared a concise update." \
  --voice-ref assets/resources/a.wav \
  --reference-text "This little work was finished in the year eighteen o three, and intended for immediate publication." \
  --request-option reference_duration_sec=5 \
  --out out.wav
```

For MeanFlow, install `dots_tts_mf_q8_0` and use
`models/DotTTS-MF-GGUF/dots-tts-mf-q8_0.gguf` with the same runtime options.

## MioTTS

MioTTS is a 1.7B voice-clone TTS path that uses MioCodec for acoustic decoding. It requires a reference voice and a MioCodec model.
Best-of-N candidate scoring can optionally use Qwen3-ASR.

| Field | Value |
|---|---|
| Family | `miotts` |
| GGUF model | `models/MioTTS-1.7B-GGUF/miotts-1.7b-q8_0.gguf` |
| Required dependency | MioCodec through `--session-option miotts.codec_model_path=<dir>` |
| Task | `tts` |
| Modes | `offline` |
| Languages | Model auto-handles supported text languages; no explicit language selector is exposed |
| Voice input | Required reference WAV through `--voice-ref` |
| Built-in voices | Not exposed |

```bash
audiocpp_cli --task tts --family miotts --model models/MioTTS-1.7B-GGUF/miotts-1.7b-q8_0.gguf --backend cuda --session-option miotts.codec_model_path=models/MioCodec-25Hz-44.1kHz-v2-GGUF/miocodec-25hz-44khz-v2-q8_0.gguf --text "Hello from MioTTS." --voice-ref assets/resources/b.wav --out out.wav
```

With best-of-N scoring, also provide a Qwen3-ASR model:

```bash
audiocpp_cli --task tts --family miotts --model models/MioTTS-1.7B-GGUF/miotts-1.7b-q8_0.gguf --backend cuda --session-option miotts.codec_model_path=models/MioCodec-25Hz-44.1kHz-v2-GGUF/miocodec-25hz-44khz-v2-q8_0.gguf --session-option miotts.best_of_n_asr_model_path=models/Qwen3-ASR-0.6B-GGUF/qwen3-asr-0.6b-q8_0.gguf --request-option miotts.best_of_n_enabled=true --request-option miotts.best_of_n=2 --text "Hello from MioTTS." --voice-ref assets/resources/b.wav --out out.wav
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
| `--session-option miotts.codec_model_path=<dir>` | directory | sibling MioCodec directory | MioCodec model used for acoustic decoding. |
| `--request-option miotts.best_of_n_enabled=true\|false` | bool | `false` | Run best-of-N candidate selection. |
| `--request-option miotts.best_of_n=<n>` | integer | session default | Generate n candidates and select by ASR scoring. |
| `--session-option miotts.best_of_n_default=<n>` | integer | `1` | Default best-of-N candidate count. |
| `--session-option miotts.best_of_n_max=<n>` | integer | `8` | Maximum best-of-N candidate count. |
| `--session-option miotts.best_of_n_language=auto\|en\|ja` | enum | `auto` | Default language used when scoring candidates. |
| `--session-option miotts.best_of_n_asr_model_path=<dir>` | directory | sibling Qwen3-ASR directory | Qwen3-ASR model used for best-of-N scoring. |

## MOSS-TTS-Local

MOSS-TTS-Local is the larger local-transformer MOSS TTS path. It supports text-only speech and optional zero-shot voice cloning through the framework speaker-reference interface. See [MOSS-TTS](models/moss_tts.md) for tokenizer layout, sampling options, cache options, and Nano details.

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

## MOSS-TTS-Nano

MOSS-TTS-Nano is the smaller MOSS TTS path. It supports text-only continuation generation and voice cloning through the framework speaker-reference interface. See [MOSS-TTS](models/moss_tts.md) for tokenizer layout, sampling options, cache options, and Local details.

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

## MagpieTTS

MagpieTTS Multilingual 357M is a multilingual TTS model with baked speaker
context prompts and a NanoCodec waveform decoder. The current package is a
standalone GGUF directory.

```bash
python3 tools/model_manager_v2.py install magpie_tts_orig

audiocpp_cli --task tts --family magpie_tts \
  --model models/MagpieTTS-Multilingual-357M-GGUF \
  --backend cuda \
  --language en \
  --text "The production coordinator reviewed the overnight audio report and sent one clear update." \
  --request-option voice_id=Sofia \
  --out out.wav
```

Use `--request-option voice_id=<name-or-index>` to select one of the baked
speaker prompts included with the package. See [MagpieTTS](models/magpie_tts.md)
for supported languages, long-form controls, and sampling options.

## NeuTTS

NeuTTS is an experimental English TTS family with built-in speaker prompts and
emotion-token control. The default package is the standalone 2E GGUF. See
[NeuTTS](models/neutts.md) for built-in voice ids, emotion options, streaming,
and full option details.

```bash
python3 tools/model_manager_v2.py install neutts

audiocpp_cli --task tts --family neutts \
  --model models/NeuTTS-2E-GGUF/neutts-2e-orig.gguf \
  --backend cuda \
  --text "The release checklist is almost complete, and the baseline run looks healthy." \
  --request-option voice_id=emily \
  --request-option emotion=neutral \
  --out out.wav
```

## OmniVoice

OmniVoice supports multilingual TTS, voice cloning, voice design, non-verbal tag tokens, long-form chunking, and chunked pseudo-streaming. See [OmniVoice](models/omnivoice.md) for the full guide.

| Field | Value |
|---|---|
| Family | `omnivoice` |
| Model directory | `models/OmniVoice` |
| Task | `tts` |
| Modes | `offline`, `streaming` |
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

Streaming voice clone:

```bash
audiocpp_cli --task tts --mode streaming --family omnivoice --model models/OmniVoice --backend cuda --text "Hello from OmniVoice." --voice-ref assets/resources/b.wav --reference-text "Some call me nature. Others call me Mother Nature. I've been here for over 4.5 billion years. 22,500 times longer than you." --text-chunk-size 160 --out stream.wav --out-dir stream_chunks
```

OmniVoice streaming is pseudo streaming: audio.cpp emits audio chunk events from text chunks and returns a merged final WAV. Upstream Python does not expose model-native streaming. For server SSE examples, options, and tag controls, see [OmniVoice](models/omnivoice.md).

## PocketTTS

PocketTTS supports built-in voices and voice cloning. The upstream project also supports exported voice states for fast reuse; the CLI surface here exposes built-in voice ids and reference WAVs.

PocketTTS language selection is a model-load option. When the model path points at the PocketTTS root, the loader uses `english` unless you pass `--load-option language=<name>`. Kyutai's normal non-English PocketTTS releases are smaller distilled language models intended for the fast PocketTTS path. The `_24l` variants are larger 24-layer, undistilled preview models that can sound better but are slower. Kyutai currently publishes French only as `french_24l`, not as a normal distilled `french` language directory, so French is not listed as a normal PocketTTS language here.

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
| `--session-option voxcpm2.mem_saver=true\|false` | bool | `false` | Use tighter graph workspaces and release MiniCPM/AudioVAE request graphs after completion to reduce resident VRAM. |
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
| Family | `higgs_audio_tts` |
| Model path | `models/Higgs-Audio-v3-TTS-4B-GGUF/higgs-audio-v3-tts-4b-q8_0.gguf` when installed through the model manager |
| Task | `tts` |
| Modes | `offline` |
| Languages | Model auto-handles supported languages |
| Voice input | Reference WAV through `--voice-ref`; transcript through `--reference-text` when known |
| Built-in voices | Not exposed |

```bash
audiocpp_cli --task tts --family higgs_audio_tts --model models/Higgs-Audio-v3-TTS-4B-GGUF/higgs-audio-v3-tts-4b-q8_0.gguf --backend cuda --text "Hello from Higgs Audio." --voice-ref assets/resources/b.wav --reference-text "Some call me nature. Others call me Mother Nature. I've been here for over 4.5 billion years. 22,500 times longer than you." --out out.wav
```

The model manager installs the Q8_0 standalone GGUF package by default:

```bash
python3 tools/model_manager_v2.py install --models-root models higgs_audio_tts_4b_q8_0
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--voice-ref` | WAV path | required | Reference speaker audio. |
| `--reference-text` | text | empty string | Transcript for reference audio. |
| `--text-chunk-size` | integer chars | `1024` | Long-form chunk size. |
| `--max-tokens` | integer | `2048` | Maximum generated AR tokens per chunk. |
| `--temperature` | float | `0.8` | AR sampling temperature. |
| `--top-k` | integer | `30` | AR top-k sampling limit. The narrower default is less prone to premature EOC than the Python client's `50`. |
| `--top-p` | float | `0.8` | AR nucleus sampling limit. The Python client's unfiltered equivalent is `1.0`. |
| `--repetition-penalty` | float | `1.1` | Accepted for Python API compatibility; Higgs audio-code sampling does not consume it. |

## Fish Audio S2 Pro

Fish Audio S2 Pro is a TTS and reference voice-clone model. See the dedicated
[Fish Audio guide](models/fish_audio.md) for multi-reference conditioning,
speaker-tagged turns, and the full option list.

| Field | Value |
|---|---|
| Family | `fish_audio` |
| Model path | `models/Fish-Audio-S2-Pro-GGUF/fish-audio-s2-pro-q8_0.gguf` when installed through the model manager |
| Task | `tts` |
| Modes | `offline` |
| Languages | Model auto-handles language; tested paths cover English and Chinese-style prompts |
| Voice input | Optional reference WAV through `--voice-ref`; transcript through `--reference-text` when known |
| Built-in voices | Not exposed |

Text-to-speech:

```bash
audiocpp_cli --task tts --family fish_audio --model models/Fish-Audio-S2-Pro-GGUF/fish-audio-s2-pro-q8_0.gguf --backend cuda --text "Hello from Fish Audio." --out out.wav
```

Reference voice clone:

```bash
audiocpp_cli --task tts --family fish_audio --model models/Fish-Audio-S2-Pro-GGUF/fish-audio-s2-pro-q8_0.gguf --backend cuda --text "The final render is ready for review." --voice-ref assets/resources/b.wav --reference-text "Some call me nature. Others call me Mother Nature. I've been here for over 4.5 billion years. 22,500 times longer than you." --out out.wav
```

Multiple reference pairs:

```bash
audiocpp_cli --task tts --family fish_audio --model models/Fish-Audio-S2-Pro-GGUF/fish-audio-s2-pro-q8_0.gguf --backend cuda --text "The review is ready, and I will check the final numbers." --request-option 'multi_reference_cond=[{"audio":"assets/resources/a.wav","text":"First reference transcript."},{"audio":"assets/resources/b.wav","text":"Second reference transcript."}]' --out out.wav
```

The model manager installs the Q8_0 standalone GGUF package by default:

```bash
python3 tools/model_manager_v2.py install --models-root models fish_audio_s2_pro_q8_0
```

## IndexTTS2

IndexTTS2 is a Chinese and English TTS model with voice cloning and expressive
emotion controls. See the dedicated [IndexTTS guide](models/index_tts.md) for
IndexTTS2, IndexTTS2.5, text normalization notes, conversion, and the full
option list.

| Field | Value |
|---|---|
| Family | `index_tts2` |
| Model directory | `models/IndexTTS-2` |
| Task | `tts`, `clon` |
| Modes | `offline` |
| Languages | `zh`, `en` |
| Voice input | Required reference WAV through `--voice-ref` |
| Built-in voices | Not exposed |

Quick start:

```bash
audiocpp_cli --task clon --family index_tts2 --model /path/to/IndexTTS-2 --backend cuda --language en --text "Hello from IndexTTS2." --voice-ref /path/to/reference.wav --out out.wav
```

## IndexTTS2.5

IndexTTS2.5 is the multilingual `index_tts2` variant selected from model
config. It adds Japanese, Spanish, and Arabic on top of Chinese and English.
See the dedicated [IndexTTS guide](models/index_tts.md) for variant details,
language notes, conversion, and the full option list.

| Field | Value |
|---|---|
| Family | `index_tts2` (the `2.5` variant is selected from the model config `version` field; no separate family) |
| Model directory | `models/IndexTTS2.5-GGUF` (default GGUF package `index_tts2_5_q8_0`; `index_tts2_5_f16` and `index_tts2_5_orig` also available) |
| Task | `tts`, `clon` |
| Modes | `offline` |
| Languages | `zh`, `en`, `ja`, `es`, `ar` |
| Voice input | Required reference WAV through `--voice-ref` |
| Built-in voices | Not exposed |

Quick start:

```bash
audiocpp_cli --task clon --family index_tts2 --model /path/to/IndexTTS2.5-GGUF --backend cuda --text "Hello from IndexTTS2.5." --voice-ref /path/to/reference.wav --out out.wav
```

## Irodori-TTS

Irodori-TTS is Japanese TTS under `--family irodori_tts`. v4 Small is the preferred GGUF-first package and supports no-reference speech, reference-conditioned speech, and instruction-based voice design in one checkpoint. The older 500M v3 and 600M v3 VoiceDesign packages remain supported for existing users. See [Irodori-TTS](models/irodori_tts.md) for v3/v4 differences, GGUF variants, options, and compatibility aliases.

## OuteTTS

OuteTTS 1.0 1B is a community model for 24 kHz TTS and voice cloning. The model manager installs the standalone Q8 GGUF package by default:

```bash
python tools/model_manager_v2.py install outetts_1_0_1b_q8_0 --models-root models
```

Quick start:

```bash
audiocpp_cli --task tts --family outetts \
  --model models/Llama-OuteTTS-1.0-1B_Q8/Llama-OuteTTS-1.0-1B_Q8.gguf \
  --backend cuda --text "Hello from OuteTTS." \
  --max-tokens 1024 --out out.wav
```

Voice clone quick start:

```bash
audiocpp_cli --task clon --family outetts \
  --model models/Llama-OuteTTS-1.0-1B_Q8/Llama-OuteTTS-1.0-1B_Q8.gguf \
  --backend cuda \
  --voice-ref reference.wav \
  --reference-text "The exact words spoken in reference.wav." \
  --request-option reference_language=en \
  --text "This sentence uses the cloned voice." \
  --max-tokens 1024 --out cloned.wav
```

See [OuteTTS community model usage](community_models/outetts.md) for cloning notes, GGUF packing, all options, and validation details.

## GLM-TTS

GLM-TTS is a community zero-shot Chinese and English speech-synthesis model.
Both the `tts` and `clon` routes require a clean reference WAV and its exact
transcript:

```bash
python tools/model_manager_v2.py install glm_tts --models-root models

audiocpp_cli --task clon --family glm_tts \
  --model models/GLM-TTS-Q8/GLM-TTS_Q8.gguf --backend cuda \
  --voice-ref reference.wav \
  --reference-text "The exact words spoken in reference.wav." \
  --text "Hello from GLM-TTS." \
  --seed 0 --out glm_tts.wav
```

See the [GLM-TTS community model guide](community_models/glm_tts.md) for
standalone GGUF packaging, controls, and validation results.

## Inflect v2

Inflect Micro v2 is a compact English offline TTS model with a native GGML
runtime. The model manager defaults to the standalone GGUF package; the
original source/conversion path remains documented in the community guide.
Inflect requires an external eSpeak-ng installation:

```bash
python3 tools/model_manager_v2.py install inflect_micro_v2_orig --models-root models

audiocpp_cli --task tts --family inflect_v2 \
  --model models/Inflect-Micro-v2-GGUF/inflect-micro-v2-orig.gguf --backend cuda \
  --text "Hello from Inflect Micro version two." \
  --request-option speaking_rate=1.0 \
  --request-option variation=0.667 \
  --seed 0 --out inflect.wav
```

See the [Inflect v2 community model guide](community_models/inflect_v2.md) for
eSpeak-ng paths, long-form behavior, source/conversion instructions, and
limitations.

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
| `--session-option supertonic.weight_type=native\|f32\|f16\|bf16\|q8_0` | enum | `native` | Weight storage type. |
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
