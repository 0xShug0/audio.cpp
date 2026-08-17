# ACE-Step

ACE-Step is wired as `--family ace_step --task gen`. It generates and edits music from text, lyrics, and optional source audio. The route controls whether source audio is ignored, optional, or required.

Common CLI shape:

```bash
audiocpp_cli --task gen --family ace_step --model models/Ace-Step1.5 --backend cuda --task-route <route> ...
```

## Model

| Field | Value |
|---|---|
| Family | `ace_step` |
| Model directory | `models/Ace-Step1.5` |
| Task | `gen` |
| Default DiT | `acestep-v15-turbo` |
| Optional DiT | `acestep-v15-xl-turbo`, `acestep-v15-xl-sft` |
| Default LM | `acestep-5Hz-lm-1.7B` |
| Prompt input | `--text` |
| Lyrics input | `--lyrics` |
| Source audio | Route-dependent through `--audio` |

## Text To Music

Generate a full song or music clip from prompt text and optional lyrics. Source audio is ignored by this route.

| Field | Value |
|---|---|
| Route | `text2music` |
| Source audio | Ignored |
| Planner | Used unless `audio_codes` are supplied |
| Duration | `--duration-seconds`; `-1` lets the planner/model choose |
| Metadata controls | Optional `bpm`, `keyscale`, `timesignature`, and `language` |

```bash
audiocpp_cli --task gen --family ace_step --model models/Ace-Step1.5 --backend cuda --task-route text2music --text "cinematic synth pop with clear vocals" --lyrics "We rise with the morning light" --duration-seconds 60 --out song.wav
```

Use the base DiT instead of turbo:

```bash
audiocpp_cli --task gen --family ace_step --model models/Ace-Step1.5 --backend cuda --task-route text2music --text "acoustic folk ballad" --lyrics "The river remembers our names" --duration-seconds 60 --load-option ace_step.dit_model_path=acestep-v15-base --out song.wav
```

## Complete

Complete or continue a source audio track. Source audio is optional: if provided, it conditions the continuation; otherwise the route behaves like a completion prompt.

| Field | Value |
|---|---|
| Route | `complete` |
| Source audio | Optional |
| Planner | Used |
| Duration | Source duration is not locked unless requested metadata or planner output controls it |
| Track classes | Optional `complete_track_classes` list changes the completion instruction |

```bash
audiocpp_cli --task gen --family ace_step --model models/Ace-Step1.5 --backend cuda --task-route complete --audio input.wav --text "finish this as a cinematic rock track" --out complete.wav
```

## Lego

Compose or transform a track from source audio plus a prompt. Source audio is required and preserved as the repaint context.

| Field | Value |
|---|---|
| Route | `lego` |
| Source audio | Required |
| Planner | Used |
| Duration | Locked to source audio |
| Repaint window | Used internally over the source context |
| Track name | Optional `track_name` changes the generated instruction |

```bash
audiocpp_cli --task gen --family ace_step --model models/Ace-Step1.5 --backend cuda --task-route lego --audio input.wav --text "generate a brighter guitar layer" --request-option track_name=guitar --out lego.wav
```

## Extract

Extract a target track from source audio. Source audio is required, and the route uses extraction-specific defaults for guidance and shift.

| Field | Value |
|---|---|
| Route | `extract` |
| Source audio | Required |
| Planner | Not used |
| Duration | Locked to source audio |
| Track name | Optional `track_name`; when omitted, the default extract instruction is used |
| Route defaults | `guidance_scale=7.0`, `shift=3.0`, planner chain-of-thought metadata disabled |

```bash
audiocpp_cli --task gen --family ace_step --model models/Ace-Step1.5 --backend cuda --task-route extract --audio song.wav --text "extract vocals" --request-option track_name=vocals --out vocals.wav
```

## Cover

Generate a cover from source audio using cover-tokenizer conditioning. Source audio is required and duration is locked to the source.

| Field | Value |
|---|---|
| Route | `cover` |
| Source audio | Required |
| Planner | Not used |
| Cover conditioning | Uses the FSQ cover tokenizer |
| Duration | Locked to source audio |

```bash
audiocpp_cli --task gen --family ace_step --model models/Ace-Step1.5 --backend cuda --task-route cover --audio source.wav --text "turn this into energetic pop vocals" --lyrics "We keep moving through the night" --out cover.wav
```

## Cover Without FSQ

Generate a cover from source audio without the FSQ cover-tokenizer conditioning path.

| Field | Value |
|---|---|
| Route | `cover-nofsq` |
| Source audio | Required |
| Planner | Not used |
| Cover conditioning | Does not use the FSQ cover tokenizer |
| Duration | Locked to source audio |

```bash
audiocpp_cli --task gen --family ace_step --model models/Ace-Step1.5 --backend cuda --task-route cover-nofsq --audio source.wav --text "make a softer acoustic cover" --lyrics "We keep moving through the night" --out cover_nofsq.wav
```

## Repaint

Replace a time span inside source audio. Source audio and a repaint window are required.

| Field | Value |
|---|---|
| Route | `repaint` |
| Source audio | Required |
| Planner | Not used |
| Duration | Locked to source audio |
| Required window | `--repaint-start`, `--repaint-end` |
| Repaint policy | `repaint_mode`, `repaint_strength`, or direct repaint injection/crossfade options |

```bash
audiocpp_cli --task gen --family ace_step --model models/Ace-Step1.5 --backend cuda --task-route repaint --audio song.wav --text "replace the middle with a brighter chorus" --repaint-start 20 --repaint-end 35 --out repaint.wav
```

## Shared Controls

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--task-route` | `text2music`, `complete`, `lego`, `extract`, `cover`, `cover-nofsq`, `repaint` | `text2music` | ACE-Step operation. |
| `--text` | text | required | Music prompt or edit instruction. |
| `--lyrics` | text | empty string | Vocal lyrics. |
| `--audio` | WAV path | route-dependent | Source audio for complete/edit/extract/cover routes. |
| `--duration-seconds` | float, `-1` for auto | `-1` | Target duration. Source-locked routes use source duration. |
| `--language` | language code | `en` | Vocal language for lyrics. |
| `--track-name` | text | empty string | Track name used by `lego` and `extract` instructions. |
| `--request-option complete_track_classes=a,b` | comma-separated text | empty list | Track classes for `complete`. |
| `--repaint-start` | seconds | required for `repaint` | Start time for repaint. |
| `--repaint-end` | seconds | required for `repaint` | End time for repaint. |
| `--repaint-mode` | `balanced`, `conservative`, `aggressive` | `balanced` | Preset repaint blending policy. |
| `--repaint-strength` | `0..1` | `0.5` | Repaint strength used by preset repaint mode. |
| `--num-inference-steps` | integer | `8` | Diffusion denoising steps. |
| `--guidance-scale` | float | `1.0`; `7.0` for `extract` unless overridden | Diffusion guidance scale. |
| `--seed` | integer | random if omitted | Generation seed. |
| `--request-option bpm=<n>` | integer | not set | Force BPM metadata; otherwise the planner chooses it when used. |
| `--request-option keyscale=<text>` | text | not set | Force key metadata; otherwise the planner chooses it when used. |
| `--request-option timesignature=<text>` | text | not set | Force time signature metadata; otherwise the planner chooses it when used. |
| `--request-option negative_prompt=<text>` | text | `NO USER INPUT` | Negative prompt. |
| `--request-option audio_codes=<text>` | ACE semantic code text | not set | Skip planner token generation and use supplied audio codes. |
| `--request-option audio_cover_strength=<float>` | float | `1.0` | Cover strength for cover/edit-style conditioning. |
| `--request-option cover_noise_strength=<float>` | float | `0.0` | Noise strength for cover conditioning. |
| `--request-option lm_temperature=<float>` | float | `0.85` | Planner sampling temperature. |
| `--request-option lm_cfg_scale=<float>` | float | `2.0` | Planner CFG scale. |
| `--request-option lm_top_k=<n>` | integer | `0` | Planner top-k; `0` disables top-k. |
| `--request-option lm_top_p=<float>` | float | `0.9` | Planner top-p. |
| `--request-option lm_repetition_penalty=<float>` | float | `1.0` | Planner repetition penalty. |
| `--request-option sampler_mode=<name>` | `euler`, `heun` | `euler` | Diffusion sampler mode. |
| `--request-option retake_seed=<n>` | integer, `-1` to clear | not set | Optional retake noise seed. |
| `--request-option retake_variance=<float>` | float | `0.0` | Retake noise mixing strength. |
| `--request-option flow_edit_morph=true\|false` | bool | `false` | Status: parsed for text2music, but not usable because the flow-edit diffusion overlay is not implemented. |
| `--request-option dcw_enabled=true\|false` | bool | `false` | Status: experimental dynamic-cfg wavelet path. Keep disabled unless validating that path. |

## Model Selection

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--load-option ace_step.dit_model_path=<dir>` | `acestep-v15-turbo`, `acestep-v15-base`, `acestep-v15-xl-turbo`, `acestep-v15-xl-sft` | `acestep-v15-turbo` | Select DiT variant inside the model root. |
| `--session-option ace_step.dit_weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | DiT weight type. |
| `--session-option ace_step.planner_weight_type=<type>` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Planner LM weight type. |
| `--session-option ace_step.mem_saver=true\|false` | bool | `false` | Release staged graph/cache state after request phases to reduce resident VRAM. Later requests may rebuild released graphs. |

ACE-Step GGUF packages are variant-specific. Use the Turbo GGUF for the default
`acestep-v15-turbo` path, and pass `--load-option ace_step.dit_model_path=acestep-v15-base`
when loading a Base GGUF package, or
`--load-option ace_step.dit_model_path=acestep-v15-xl-turbo` for the XL Turbo one.

### XL variants

`acestep-v15-xl-turbo` and `acestep-v15-xl-sft` are the larger DiT: 32 layers of
2560 against turbo's 24 of 2048, with 32 attention heads of 128 (so the attention
width is 4096, wider than the model). The condition encoder, audio tokenizer and
detokenizer stay at 2048 — the `encoder_hidden_size` group in the XL config — and
the DiT's condition embedder bridges the two. The XL timbre encoder also prepends
a CLS token to the reference frames and reads that position back, where earlier
variants read the first audio frame.

Both are **optional package resources**: they are only loadable when their
weights are present, and a package without them loads and runs exactly as
before. Selecting one that is not installed reports which directory is missing.
The upstream snapshots ship four safetensors shards plus a
`model.safetensors.index.json`, which the package spec points at directly.

The two differ only in `is_turbo`: XL Turbo is guidance-distilled and ignores
`guidance_scale`, XL SFT takes the CFG path the way `acestep-v15-base` does.
Their dimensions, encoder group and head configuration are identical.

`ace_step_xl_turbo_bf16` and `ace_step_xl_sft_bf16` install them as GGUFs
(14.2 GB each), self-contained the way the Turbo and Base GGUFs are — XL DiT,
planner LM, text encoder and VAE in one file:

```bash
audiocpp_cli --task gen --family ace_step --model models/ACE-Step1.5-GGUF/xl-turbo --backend cuda --task-route text2music --text "warm lo-fi hip hop with a soft rhodes piano" --duration-seconds 60 --load-option ace_step.dit_model_path=acestep-v15-xl-turbo --out song.wav
```

Running one from a safetensors tree instead is worth a `dit_weight_type=bf16`,
because the XL snapshots are stored in float32 and `native` puts 19.9 GB of
weights on the card:

```bash
audiocpp_cli --task gen --family ace_step --model models/Ace-Step1.5 --backend cuda --task-route text2music --text "warm lo-fi hip hop with a soft rhodes piano" --duration-seconds 60 --load-option ace_step.dit_model_path=acestep-v15-xl-sft --session-option ace_step.dit_weight_type=bf16 --out song.wav
```

Measured on an RTX 5090, 20 s of audio, weight loading included and the weights
warm in the page cache: 87 s from safetensors at `native`, 25 s from safetensors
at `bf16`, 15 s from the bf16 GGUF, both variants alike (turbo, for reference:
11 s). Reading the weights off disk adds roughly 10 s either way.

Building an XL GGUF yourself needs the other variants' safetensors on hand,
because `audiocpp_gguf` checks the conversion against the spec's required
namespaces; exclude them from the output:

```bash
audiocpp_gguf --root models/Ace-Step1.5 --family ace_step \
  --input dit_turbo_weights=models/Ace-Step1.5/acestep-v15-turbo/model.safetensors \
  --input dit_turbo_silence_latent=models/Ace-Step1.5/acestep-v15-turbo/silence_latent.safetensors \
  --input dit_base_weights=models/Ace-Step1.5/acestep-v15-base/model.safetensors \
  --input dit_base_silence_latent=models/Ace-Step1.5/acestep-v15-base/silence_latent.safetensors \
  --input dit_xl_turbo_weights=models/Ace-Step1.5/acestep-v15-xl-turbo/model.safetensors.index.json \
  --input dit_xl_turbo_silence_latent=models/Ace-Step1.5/acestep-v15-xl-turbo/silence_latent.safetensors \
  --input lm_weights=models/Ace-Step1.5/acestep-5Hz-lm-1.7B/model.safetensors \
  --input text_encoder_weights=models/Ace-Step1.5/Qwen3-Embedding-0.6B/model.safetensors \
  --input vae_weights=models/Ace-Step1.5/vae/diffusion_pytorch_model.safetensors \
  --exclude-prefix dit_turbo_ --exclude-prefix dit_base_ \
  --type bf16 --output ace-step-1.5-xl-turbo-bf16.gguf
```

Swap `dit_xl_turbo_*` for `dit_xl_sft_*` to build the SFT one. Upstream ships
`silence_latent.pt` where the spec wants safetensors;
`tests/ace_step/convert_silence_latent.py --input <variant>/silence_latent.pt`
converts it.
