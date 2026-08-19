# Music, Video, And Sound Generation

| Model | Family | Main Route(s) | Quick Start |
|---|---|---|---|
| ACE-Step | `ace_step` | text-to-music, edit, cover, repaint | [ACE-Step](#ace-step) |
| MiniMax-H3 | `minimax_h3` | text-to-audio, dialogue, video | [MiniMax-H3](#minimax-h3) |
| MiniMax Music 3 | `minimax_music3` | text-to-music, lyrics conditioning | [MiniMax Music 3](#minimax-music-3) |
| Stable Audio | `stable_audio` | music, SFX, init-audio, inpaint | [Stable Audio](#stable-audio) |
| HeartMuLa | `heartmula` | lyrics/tags to music | [HeartMuLa](#heartmula) |

Use `--task gen` for models that generate music, sound effects, video, or audio from text and optional audio conditioning. These models are not normal TTS models: text chunking for speech TTS does not apply unless a model explicitly documents a long-output mode.

Common CLI shape:

```bash
audiocpp_cli --task gen --family <family> --model <model-dir> --backend cuda ...
```

## ACE-Step

ACE-Step generates and edits music from prompts, lyrics, and optional source audio. See [ace_step.md](models/ace_step.md) for the full route manual.

| Field | Value |
|---|---|
| Family | `ace_step` |
| Model directory | `models/Ace-Step1.5` |
| Task | `gen` |
| Routes | `text2music`, `complete`, `lego`, `extract`, `cover`, `cover-nofsq`, `repaint` |
| Main inputs | Prompt text, optional lyrics, optional source audio depending on route |
| Languages | 19+ lyric languages supported by the model |

```bash
audiocpp_cli --task gen --family ace_step --model models/Ace-Step1.5 --backend cuda --task-route text2music --text "cinematic synth pop with clear vocals" --lyrics "We rise with the morning light" --duration-seconds 60 --out song.wav
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--task-route` | `text2music`, `complete`, `lego`, `extract`, `cover`, `cover-nofsq`, `repaint` | `text2music` | ACE-Step operation. |
| `--text` | text | required | Music prompt or edit instruction. |
| `--lyrics` | text | empty string | Vocal lyrics. |
| `--audio` | WAV path | not set | Source audio for edit/extract/cover routes. |
| `--duration-seconds` | float, `-1` for auto | `-1` | Target duration. |
| `--num-inference-steps` | integer | `8` | Diffusion denoising steps. |
| `--guidance-scale` | float | `1.0` | Diffusion guidance scale. |
| `--session-option ace_step.mem_saver=true\|false` | bool | `false` | Release staged graph/cache state after request phases to reduce resident VRAM. Later requests may rebuild released graphs. |

## MiniMax-H3

MiniMax-H3 generates prompt-conditioned audio and optional video. It can be used for music-like audio, dialogue-style audio, and text-to-video experiments through the same `gen` route. See [MiniMax-H3](community_models/minimax_h3.md) for component GGUF package layout, conversion, options, and performance notes.

Audio-only prompt:

```bash
audiocpp_cli --task gen --family minimax_h3 \
  --model models/MiniMax-H3-Q4-GGUF/dit.gguf \
  --backend cuda \
  --text "A lively four-speaker podcast scene with clear voices, quick timing, and no background music." \
  --num-inference-steps 20 \
  --guidance-scale 1.0 \
  --request-option height=32 \
  --request-option width=32 \
  --request-option num_frames=481 \
  --request-option return_video=false \
  --out output.wav
```

Set `--request-option return_video=true` and provide `--out-dir` when you also want video frames as an output artifact.

## MiniMax Music 3

MiniMax Music 3 generates songs from a detailed music caption plus lyrics. See
[MiniMax Music 3](community_models/minimax_music3.md) for package layout,
component GGUF selection, options, and current performance/memory notes.

```bash
CAPTION='A bright pop rock song with clean drums, crisp rhythm guitars, a clear female vocal, an energetic chorus, and polished studio production.'
LYRICS='[verse] City lights are shining low. I keep moving with the glow. [chorus] Turn it up and let it fly. Sing the melody tonight.'

audiocpp_cli --task gen --family minimax_music3 \
  --model models/MiniMax-Music3-GGUF \
  --backend cuda \
  --text "$CAPTION" \
  --request-option "lyrics=$LYRICS" \
  --request-option duration_sec=30 \
  --request-option num_inference_steps=30 \
  --out music3.wav
```

`duration_sec` sets the AR frame budget rather than a strict final audio length.
Larger values can produce longer songs and also increase VRAM use.

## Stable Audio

Stable Audio generates music or sound effects from text. It can also use source audio for init-audio or inpainting workflows. See [stable_audio.md](models/stable_audio.md) for the full Stable Audio manual.

| Field | Value |
|---|---|
| Family | `stable_audio` |
| Model directories | `models/stable-audio-3-small-music`, `models/stable-audio-3-small-sfx`, `models/stable-audio-3-medium` |
| Task | `gen` |
| Modes | Text-to-music, text-to-SFX, init-audio, inpainting |
| Main inputs | Prompt text; optional source audio for audio-conditioned modes |
| Languages | `en` prompts |

Text to music:

```bash
audiocpp_cli --task gen --family stable_audio --model models/stable-audio-3-small-music --backend cuda --text "uplifting house music with bright synths and festival drums" --duration-seconds 30 --out music.wav
```

Inpainting:

```bash
audiocpp_cli --task gen --family stable_audio --model models/stable-audio-3-small-music --backend cuda --audio input.wav --text "replace the masked sections with a tight snare fill" --duration-seconds 10 --request-option audio_input_kind=inpaint_audio --request-option inpaint_mask_start_seconds=2.5 --request-option inpaint_mask_end_seconds=3.5 --out inpaint.wav
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--text` | prompt text | required | Music or sound-effect prompt. |
| `--audio` | WAV path | not set | Source audio for init-audio or inpainting. |
| `--duration-seconds` | `seconds[,seconds...]` | `120` | Target duration per prompt. Use one value for all prompts, or one comma-separated value per Stable Audio `batch_size` item. |
| `--num-inference-steps` | integer | `8` | RF diffusion steps. |
| `--guidance-scale` | float | `1.0` | Classifier-free guidance scale. |
| `--request-option audio_input_kind=<kind>` | `init_audio`, `inpaint_audio` | `init_audio` when `--audio` is provided | How the source audio is used. |
| `--request-option init_noise_level=<float>` | `0..1` | `1.0` | Strength for init-audio conditioning. |
| `--request-option inpaint_mask_start_seconds=<list>` | comma-separated seconds | not set | Inpaint region start times. |
| `--request-option inpaint_mask_end_seconds=<list>` | comma-separated seconds | not set | Inpaint region end times. |
| `--session-option stable_audio.mem_saver=true\|false` | bool | `false` | Release staged graph/cache state after request phases to reduce resident VRAM. Later requests may rebuild released graphs. |

## HeartMuLa

HeartMuLa generates music from lyrics and tags. The upstream reference treats tags as style/control text: it lowercases them and wraps them with special tag tokens internally. The audio.cpp integration follows that behavior and does not use speaker-reference audio.

| Field | Value |
|---|---|
| Family | `heartmula` |
| Model directory | `models/HeartMuLa` |
| Task | `gen` |
| Modes | Lyrics/tags to music; optional infinite mode for longer outputs |
| Main inputs | `--lyrics` plus comma-separated `tags` |
| Languages | Multilingual lyrics supported by the model |
| Reference audio | Not consumed by this integration |
| Tag format | Free-form comma-separated descriptors such as genre, mood, instruments, tempo, and vocals |

```bash
audiocpp_cli --task gen --family heartmula --model models/HeartMuLa --backend cuda --text "a bright pop chorus with drums" --lyrics "We rise with the morning light" --request-option tags=pop,bright,drums --out song.wav
```

Long output mode:

```bash
audiocpp_cli --task gen --family heartmula --model models/HeartMuLa --backend cuda --text "long cinematic pop song" --lyrics "Verse one begins with a quiet street. The chorus opens wide." --request-option tags=pop,cinematic,drums,vocal --duration-seconds 300 --request-option infinite_mode=true --out song_long.wav
```

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--text` | text | required | Music prompt or short description. |
| `--lyrics` | text | empty string | Lyrics for generated music. |
| `--request-option tags=<text>` | comma-separated text | required | Music tags; the model path wraps them as tag tokens internally. |
| `--request-option duration_sec=<seconds>` | seconds | `120` | Maximum generated duration. |
| `--temperature` | float | `1.0` | Music-token sampling temperature. |
| `--top-k` | integer | `50` | Music-token top-k sampling limit. |
| `--guidance-scale` | float | `1.5` | MuLa classifier-free guidance scale. |
| `--num-inference-steps` | integer | `10` | Codec flow solver steps. |
| `--request-option codec_duration_sec=<seconds>` | seconds | `29.76` | Codec detokenization chunk duration. |
| `--request-option codec_guidance_scale=<float>` | float | `1.25` | Codec classifier-free guidance scale. |
| `--request-option infinite_mode=true\|false` | bool | `false` | Generate long outputs by splitting lyrics into bounded HeartMuLa requests. |
| `--text-chunk-size` | chars | `4096` | Text chunk size for infinite mode. |
| `--request-option infinite_chunk_audio_duration_ms=<n>` | milliseconds | `240000` | Per-chunk audio cap for infinite mode. |
| `--seed` | integer | `1234` | Generation seed. |
| `--session-option heartmula.weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | MuLa and codec weight storage type. |
| `--session-option heartmula.generator_weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `heartmula.weight_type` or `native` | MuLa music-token generator weight storage type. |
| `--session-option heartmula.codec_weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `heartmula.weight_type` or `native` | Codec weight storage type. |
| `--session-option heartmula.mem_saver=true\|false` | bool | `false` | Release staged graph/cache state after AR/codec phases and infinite-mode chunks to reduce resident VRAM. Later requests may rebuild released graphs. |

Compatibility mapping:

| Legacy option | Schema-v1 option |
|---|---|
| `duration_seconds` | `duration_sec` |
| `codec_duration` | `codec_duration_sec` |
| `infinite_chunk_audio_length_ms` | `infinite_chunk_audio_duration_ms` |
| `heartmula.mula_weight_type` | `heartmula.generator_weight_type` |
| `heartmula.mula_weight_context_mb` | `heartmula.generator_weight_context_mb` |
| `heartmula.mula_constant_context_mb` | `heartmula.generator_constant_context_mb` |
| `heartmula.mula_backbone_prefill_graph_arena_mb` | `heartmula.backbone_prefill_graph_arena_mb` |
| `heartmula.mula_backbone_step_graph_arena_mb` | `heartmula.backbone_step_graph_arena_mb` |
| `heartmula.mula_decoder_prefill_graph_arena_mb` | `heartmula.decoder_prefill_graph_arena_mb` |
| `heartmula.mula_decoder_step_graph_arena_mb` | `heartmula.decoder_step_graph_arena_mb` |
| `heartmula.mula_frame_embedding_graph_arena_mb` | `heartmula.frame_embedding_graph_arena_mb` |
| `heartmula.codec_flow_estimator_graph_arena_mb` | `heartmula.flow_estimator_graph_arena_mb` |
| `heartmula.codec_conditioning_graph_arena_mb` | `heartmula.conditioning_graph_arena_mb` |
| `heartmula.codec_scalar_decoder_graph_arena_mb` | `heartmula.scalar_decoder_graph_arena_mb` |

For the full backend memory-arena controls, use `audiocpp_cli --help --model <model-dir> --family heartmula`.
