# Stable Audio

Stable Audio is wired as `--family stable_audio --task gen`. It generates music or sound effects from text, and the music model can condition generation on input audio for init-audio or inpainting workflows.

## Stable Audio 3 Small Music

This is the default music-generation package. Use it for text-to-music, init-audio, and inpainting.

| Field | Value |
|---|---|
| Family | `stable_audio` |
| Model directory | `models/stable-audio-3-small-music` |
| Task | `gen` |
| Main modes | Text-to-music, init-audio, inpainting |
| Prompt language | `en` |
| Optional audio input | `--audio` with `audio_input_kind=init_audio` or `audio_input_kind=inpaint_audio` |

Text to music:

```bash
audiocpp_cli --task gen --family stable_audio --model models/stable-audio-3-small-music --backend cuda --text "uplifting house music with bright synths and festival drums" --duration-seconds 30 --out music.wav
```

Init audio:

```bash
audiocpp_cli --task gen --family stable_audio --model models/stable-audio-3-small-music --backend cuda --audio input.wav --text "turn this into mellow downtempo music with deeper bass" --duration-seconds 10 --request-option audio_input_kind=init_audio --request-option init_noise_level=0.45 --out init_audio.wav
```

Inpaint audio:

```bash
audiocpp_cli --task gen --family stable_audio --model models/stable-audio-3-small-music --backend cuda --audio input.wav --text "replace the masked sections with a tight snare fill and bright guitar swell" --duration-seconds 10 --request-option audio_input_kind=inpaint_audio --request-option inpaint_mask_start_seconds=2.5,6.5 --request-option inpaint_mask_end_seconds=3.5,8.0 --out inpaint.wav
```

## Stable Audio 3 Small SFX

Use this package for short sound effects. It uses the same `gen` task but is intended for SFX prompts instead of music prompts.

| Field | Value |
|---|---|
| Family | `stable_audio` |
| Model directory | `models/stable-audio-3-small-sfx` |
| Task | `gen` |
| Main mode | Text-to-sound-effects |
| Prompt language | `en` |
| Optional audio input | Not part of the basic SFX command |

```bash
audiocpp_cli --task gen --family stable_audio --model models/stable-audio-3-small-sfx --backend cuda --text "footsteps on gravel, close perspective, crisp natural stone texture" --duration-seconds 8 --out sfx.wav
```

## Stable Audio 3 Medium

Use the medium package for the larger Stable Audio 3 music model. It uses the same user-facing controls as the music path.

| Field | Value |
|---|---|
| Family | `stable_audio` |
| Model directory | `models/stable-audio-3-medium` |
| Task | `gen` |
| Main mode | Text-to-music |
| Prompt language | `en` |
| Optional audio input | Supported by the Stable Audio request path when the model package provides the needed components |

```bash
audiocpp_cli --task gen --family stable_audio --model models/stable-audio-3-medium --backend cuda --text "wide cinematic ambient music with soft piano and evolving strings" --duration-seconds 30 --out music_medium.wav
```

## Common Options (use directly)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `--text` | prompt text | required | Music or sound-effect prompt. |
| `--audio` | WAV path | not set | Source audio for init-audio or inpainting. |
| `--duration-seconds` | `seconds[,seconds...]` | `120` | Target duration per prompt. Use one value for all prompts, or one comma-separated value per Stable Audio `batch_size` item. |
| `--num-inference-steps` | integer | `8` | RF diffusion steps. |
| `--guidance-scale` | float | `1.0` | Classifier-free guidance scale. |
| `--seed` | integer | random if omitted | Generation seed. |

## Request Options (use with `--request-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `prompt` | text | not set | Fallback text prompt when text_input is absent; pipe separates batch prompts. |
| `negative_prompt` | text | not set | Negative prompt. |
| `duration_sec` | seconds or comma-separated seconds | `120` | Target duration per batch prompt; encode multiple durations as a comma-separated string, not a JSON array. |
| `batch_size` | integer | `1` | Prompt batch size. |
| `num_inference_steps` | integer >= 1 | `8` | Diffusion steps. |
| `guidance_scale` | float | `1` | Classifier-free guidance scale. |
| `apg_scale` | float | `1` | Adaptive projected guidance scale. |
| `sampler` | `pingpong`, `euler`, `dpmpp-2m`, `dpmpp-3m-sde` | not set | Diffusion sampler for Stable Audio 3. The foundation/medium path also accepts the DPM++ samplers. |
| `seed` | integer >= 0 | random | Sampling seed; omitted chooses a random seed. |
| `duration_padding_sec` | seconds | `6` | Extra generated padding before truncation. |
| `truncate_output_to_duration` | bool | `true` | Trim decoded audio to requested duration. |
| `chunked_decode` | bool | `true` | Decode the autoencoder in chunks. |
| `init_noise_level` | `0..1` | `1` | Strength for audio-conditioned generation. |
| `audio_input_kind` | `init_audio`, `inpaint_audio` | `init_audio` | How the model uses input audio. |
| `inpaint_mask_start_sec` | comma-separated seconds | not set | Inpaint region start times. |
| `inpaint_mask_end_sec` | comma-separated seconds | not set | Inpaint region end times. |
| `sigma_min` | float | not set | Foundation sampler minimum noise level. |
| `sigma_max` | float | not set | Foundation sampler maximum noise level. |
| `rho` | float | not set | Foundation sampler noise schedule exponent. |

The corresponding legacy `*_seconds` request names remain accepted. Use only
one name for each control in a request.

## Session Options (use with `--session-option`)

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `stable_audio.max_batch` | integer >= 1 | `1` | Maximum prompt batch size. |
| `stable_audio.weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | not set | Weight storage type; defaults to native. |
| `stable_audio.mem_saver` | bool | `false` | Release staged graph/cache state after conditioner, diffusion, and autoencoder phases to reduce resident VRAM. Later requests may rebuild released graphs. |

For backend weight-type controls, use `audiocpp_cli --inspect --model <model-dir> --family stable_audio`.
